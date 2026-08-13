from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from .config import SimulationConfig
from .domain import OffloadAction, OffloadEstimate, OffloadResult, Task, VehicleState
from .dqn import DQNAgent
from .metrics import MetricsAccumulator, RunRecorder, RunSummary
from .mobility import create_mobility
from .network import (
    CompactLinkThroughputMap,
    ReverseParetoV2VIndex,
    ServiceSpatialIndex,
    build_compact_neighbor_graph,
    cloud_queue_delay_s,
    distance_m,
    should_build_reverse_pareto_index,
    transmission_delay_s,
)
from .pricing import (
    LeaderPriceEvaluation,
    cloud_leader_price,
    cloud_price,
    cloud_price_candidates,
    evaluate_cloud_leader_response,
    service_quote,
)
from .routes import prepare_sumo_scenario
from .serverless import AnalyticalServerlessBackend, HttpKnativeBackend, ServerlessBackend
from .strategies import (
    DecisionContext,
    choose_action,
    decision_state,
    estimates,
    game_guidance,
    hybrid_arbitration,
    hybrid_decision_source,
    offload_preprocessing,
    policy_action_ids,
    reward_for,
    stackelberg_best_action,
    v2i_estimate,
)


@dataclass(slots=True)
class _BatchFollowerResponse:
    cloud_cycles: float
    late_tasks: float
    cloud_requests: float
    action_probabilities: np.ndarray
    iterations: int = 0
    request_residual: float = math.inf
    cycle_residual: float = math.inf


@dataclass(slots=True)
class _PreparedFollowerResponse:
    cycles: np.ndarray
    deadlines: np.ndarray
    delays: np.ndarray
    allowed: np.ndarray
    game_actions: np.ndarray
    confidence: np.ndarray
    dominates: np.ndarray
    anticipated_capacity_ratio: float
    states: np.ndarray | None


class _DecisionTrace:
    """Compact deterministic action trace used by open-loop backend replay."""

    def __init__(self, config: SimulationConfig):
        self.mode = config.decision_trace_mode
        self.path = (
            Path(config.decision_trace_path).resolve()
            if config.decision_trace_path
            else None
        )
        self.handle = None
        if self.mode == "record":
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = gzip.open(self.path, "wt", encoding="utf-8", newline="\n")
        elif self.mode == "replay":
            assert self.path is not None
            self.handle = gzip.open(self.path, "rt", encoding="utf-8")

    def apply(self, step: int, decisions: list[dict]) -> None:
        if self.mode == "none":
            return
        assert self.handle is not None
        if self.mode == "record":
            for decision in decisions:
                action = decision["action"]
                candidate = decision["candidate_map"][action]
                self.handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "task_id": decision["task"].task_id,
                            "action": int(action),
                            "target_id": candidate.target_id,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            return
        for decision in decisions:
            line = self.handle.readline()
            if not line:
                raise RuntimeError("decision replay trace ended before the simulation")
            saved = json.loads(line)
            task = decision["task"]
            if int(saved["step"]) != step or saved["task_id"] != task.task_id:
                raise RuntimeError(
                    "decision replay task mismatch: "
                    f"expected step={step} task={task.task_id}, received {saved}"
                )
            action = OffloadAction(int(saved["action"]))
            candidate = decision["candidate_map"][action]
            if action == OffloadAction.V2V and candidate.target_id != saved["target_id"]:
                raise RuntimeError(
                    "decision replay V2V target changed: "
                    f"task={task.task_id} expected={saved['target_id']} "
                    f"received={candidate.target_id}"
                )
            decision["action"] = action

    def close(self, completed: bool) -> None:
        if self.handle is None:
            return
        try:
            if completed and self.mode == "replay":
                extra = self.handle.readline()
                if extra:
                    raise RuntimeError("decision replay trace contains unused records")
        finally:
            self.handle.close()


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.mobility == "sumo" and config.scenario_net:
            config.scenario_config = str(
                prepare_sumo_scenario(
                    config.scenario_net,
                    config.route_output_dir,
                    config.vehicle_count,
                    config.route_departure_end_s,
                    config.seed,
                )
            )
        self.environment_rng = random.Random(config.seed)
        self.policy_rng = random.Random(config.seed + 1)
        self.dqn = DQNAgent(config.dqn, config.seed + 2)
        if config.dqn.mode == "evaluate":
            self.dqn.load(config.dqn.checkpoint_path)
        self.mobility = create_mobility(config)
        self.backend = self._create_backend()
        self.vehicle_energy: dict[str, float] = {}
        self.service_workload_cycles: dict[str, float] = {}
        self.workload_compute_hz: dict[str, float] = {}
        self.cloud_workload_cycles = 0.0
        self.price_history: list[float] = []
        self.current_cloud_price = config.cloud_base_price
        self.last_price_response_diagnostics: dict[str, float | int] = {}
        self.last_cloud_offload_ratio = config.cloud_target_offload_ratio
        self.decision_trace = _DecisionTrace(config)
        # Decayed counts of DQN overrides that flipped task success relative to
        # the game action's decision-time estimate.  Neutral overrides do not
        # move the estimate, so learned congestion avoidance is not penalized.
        self._override_beneficial_weight = 0.0
        self._override_harmful_weight = 0.0
        # Decayed outcomes of decisions that followed the game action.  Their
        # realized success rate is the game's demonstrated adequacy A: the
        # refutation defense is trusted only while the game is near-perfect,
        # and the game evidence itself is damped as A degrades.
        self._game_follow_success_weight = 0.0
        self._game_follow_failure_weight = 0.0

    @staticmethod
    def _mode_active(mode: str, dqn_mode: str) -> bool:
        if mode == "always":
            return True
        return mode == "evaluate" and dqn_mode == "evaluate"

    def _online_reliability_active(self) -> bool:
        return self._mode_active(
            self.config.decision.hybrid_online_reliability,
            self.config.dqn.mode,
        )

    def _adequacy_active(self) -> bool:
        return self._mode_active(
            self.config.decision.hybrid_game_adequacy_arbitration,
            self.config.dqn.mode,
        )

    def _hybrid_online_reliability(self) -> float:
        if not self._online_reliability_active():
            return 1.0
        reliability = (1.0 + self._override_beneficial_weight) / (
            1.0
            + self._override_beneficial_weight
            + self._override_harmful_weight
        )
        reliability = max(
            self.config.decision.hybrid_reliability_floor, reliability
        )
        if not self._adequacy_active():
            return reliability
        # Blend the defense out as the game's demonstrated adequacy drops:
        # a refuted DQN only stays suppressed while following the game is
        # empirically sufficient.
        weight = self._adequacy_defense_weight(self._game_adequacy())
        return 1.0 - weight * (1.0 - reliability)

    def _game_adequacy(self) -> float:
        return (1.0 + self._game_follow_success_weight) / (
            1.0
            + self._game_follow_success_weight
            + self._game_follow_failure_weight
        )

    def _adequacy_defense_weight(self, adequacy: float) -> float:
        decision = self.config.decision
        floor = decision.hybrid_adequacy_defense_floor
        full = decision.hybrid_adequacy_defense_full
        if adequacy >= full:
            return 1.0
        if adequacy <= floor or full <= floor:
            return 0.0
        return (adequacy - floor) / (full - floor)

    def _arbitration_game_adequacy(self) -> float:
        """Return A for game-evidence damping, or one when inactive."""
        if not self._adequacy_active():
            return 1.0
        return self._game_adequacy()

    def _decay_override_outcomes(self) -> None:
        if not (self._online_reliability_active() or self._adequacy_active()):
            return
        decay = self.config.decision.hybrid_reliability_decay
        self._override_beneficial_weight *= decay
        self._override_harmful_weight *= decay
        self._game_follow_success_weight *= decay
        self._game_follow_failure_weight *= decay

    def _record_override_outcome(
        self,
        decision: dict,
        success: bool,
    ) -> None:
        if not (self._online_reliability_active() or self._adequacy_active()):
            return
        if self.config.strategy != "hybrid_stackelberg":
            return
        baseline_action = decision["baseline_action"]
        if decision["action"] == baseline_action:
            if success:
                self._game_follow_success_weight += 1.0
            else:
                self._game_follow_failure_weight += 1.0
            return
        if not decision["used_dqn"]:
            return
        task = decision["task"]
        baseline_estimate = decision["candidate_map"][baseline_action]
        baseline_on_time = baseline_estimate.delay_s <= task.deadline_s
        if success and not baseline_on_time:
            self._override_beneficial_weight += 1.0
        elif not success and baseline_on_time:
            self._override_harmful_weight += 1.0

    def run(self) -> tuple[RunSummary, str]:
        run_started = perf_counter()
        phase_seconds = {
            "mobility": 0.0,
            "task_setup": 0.0,
            "topology_and_pricing": 0.0,
            "estimation": 0.0,
            "policy": 0.0,
            "execution_and_state": 0.0,
            "dqn_training": 0.0,
            "metrics_and_logging": 0.0,
        }
        segment_size = 250
        segments = [
            {
                "start_step": start,
                "end_step": min(start + segment_size, self.config.steps) - 1,
                "wall_s": 0.0,
                "tasks": 0,
                "peak_active_vehicles": 0,
            }
            for start in range(0, self.config.steps, segment_size)
        ]
        recorder = RunRecorder(self.config.output_dir, self.config)
        accumulator = MetricsAccumulator(self.config.vehicle_compute_hz)
        departed: set[str] = set()
        peak_active = 0
        cloud_queue = 0
        pending_transitions: dict[str, tuple] = {}
        completed_steps = 0
        mobility_started = False
        completed_trace = False
        try:
            self.mobility.start()
            mobility_started = True
            for step in range(self.config.steps):
                step_started = perf_counter()
                self._decay_override_outcomes()
                self._advance_workloads()
                cloud_queue = self._cloud_backlog_tasks()
                phase_started = perf_counter()
                frame = self.mobility.step(step)
                phase_seconds["mobility"] += perf_counter() - phase_started
                completed_steps = step + 1
                departed.update(frame.departed_ids)
                peak_active = max(peak_active, len(frame.vehicles))
                phase_started = perf_counter()
                vehicles = self._vehicle_states(frame.vehicles)
                if not vehicles:
                    cloud_queue = 0
                    segments[step // segment_size]["wall_s"] += perf_counter() - step_started
                    continue
                if self.config.service_role_mode == "fixed_ratio":
                    for vehicle in vehicles.values():
                        vehicle.is_service = self._is_fixed_service_vehicle(vehicle.vehicle_id)
                    tasks = self._generate_tasks(step, vehicles)
                    task_vehicle_ids = {task.vehicle_id for task in tasks}
                else:
                    tasks = self._generate_tasks(step, vehicles)
                    task_vehicle_ids = {task.vehicle_id for task in tasks}
                    for vehicle in vehicles.values():
                        vehicle.is_service = vehicle.vehicle_id not in task_vehicle_ids
                for vehicle in vehicles.values():
                    # A service vehicle is an ordinary 2 GHz vehicle that is idle
                    # in the current step, not a faster node with a role-dependent CPU.
                    vehicle.compute_hz = self.config.vehicle_compute_hz
                    vehicle.queue_length = math.ceil(
                        vehicle.workload_cycles / max(vehicle.compute_hz, 1.0)
                    )
                    if vehicle.workload_cycles > 0:
                        self.workload_compute_hz[vehicle.vehicle_id] = vehicle.compute_hz
                accumulator.observe_step(
                    active_vehicle_count=len(vehicles),
                    task_vehicle_count=len(tasks),
                    service_vehicle_count=sum(item.is_service for item in vehicles.values()),
                    arrived_cycles=sum(task.compute_cycles for task in tasks),
                    step=step,
                )
                phase_seconds["task_setup"] += perf_counter() - phase_started
                phase_started = perf_counter()
                adjacency = build_compact_neighbor_graph(
                    vehicles,
                    self.config.network.neighbor_radius_m,
                    self.config.network,
                )
                scaled_cloud_queue = self._scaled_cloud_queue(cloud_queue)
                cloud_target_offload_ratio = self._cloud_capacity_target(tasks)
                if self.config.cloud_pricing_mode in {
                    "leader_best_response",
                    "follower_best_response",
                } and (
                    self.config.cloud_pricing_mode != "follower_best_response"
                    or self.config.strategy in {"stackelberg", "hybrid_stackelberg"}
                ):
                    current_cloud_price = cloud_leader_price(
                        self.current_cloud_price,
                        self.last_cloud_offload_ratio,
                        self.config.cloud_base_price,
                        cloud_target_offload_ratio,
                        self.config.cloud_demand_sensitivity,
                        self.config.cloud_price_smoothing,
                        scaled_cloud_queue,
                        self.config.network.queue_delay_threshold,
                        self._cloud_utilization_ratio(),
                        self.config.serverless.capacity_utilization_target,
                        self.config.cloud_capacity_price_weight,
                        self.config.cloud_min_price,
                        self.config.cloud_max_price,
                    )
                else:
                    current_cloud_price = cloud_price(
                        self.config.cloud_base_price,
                        scaled_cloud_queue,
                        self.config.network.queue_delay_threshold,
                    )
                quote_by_id = {
                    quote.vehicle_id: quote
                    for vehicle in vehicles.values()
                    if vehicle.is_service
                    for quote in [
                        service_quote(
                            vehicle,
                            current_cloud_price,
                            self.config.service_price_sensitivity,
                            self.config.service_min_energy,
                            self.config.service_max_queue,
                        )
                    ]
                    if quote is not None
                }
                throughput_by_link = CompactLinkThroughputMap(adjacency)
                service_spatial_index = ServiceSpatialIndex(vehicles, quote_by_id)
                service_spatial_index.prepare_sources(
                    task_vehicle_ids,
                    self.config.network.neighbor_radius_m,
                )
                v2v_path_index = (
                    ReverseParetoV2VIndex(
                        adjacency,
                        vehicles,
                        quote_by_id,
                        self.config.network.max_hops,
                        self.config.service_compute_hz,
                        self.config.energy,
                    )
                    if should_build_reverse_pareto_index(
                        len(vehicles),
                        len(quote_by_id),
                        len(tasks),
                    )
                    else None
                )
                phase_seconds["topology_and_pricing"] += perf_counter() - phase_started
                batch_http = isinstance(self.backend, HttpKnativeBackend)
                step_http_overhead_s = (
                    self.backend.predicted_platform_overhead_s() if batch_http else None
                )
                snapshot_max_service_workload_s = max(
                    (
                        item.workload_cycles / max(item.compute_hz, 1.0)
                        for item in vehicles.values()
                        if item.is_service
                    ),
                    default=0.0,
                )
                decisions: list[dict] = []
                transitions_to_store: list[tuple] = []
                learned_strategy = self.config.strategy in {
                    "dqn",
                    "hybrid_stackelberg",
                }
                for task in tasks:
                    phase_started = perf_counter()
                    vehicle = vehicles[task.vehicle_id]
                    source_workload_s = vehicle.workload_cycles / max(vehicle.compute_hz, 1.0)
                    server_position = min(
                        self.config.service_positions,
                        key=lambda position: (
                            (vehicle.position[0] - position[0]) ** 2
                            + (vehicle.position[1] - position[1]) ** 2,
                            position,
                        ),
                    )
                    server_distance = distance_m(vehicle.position, server_position)
                    context = DecisionContext(
                        task=task,
                        vehicle=vehicle,
                        vehicles=vehicles,
                        service_quotes=quote_by_id,
                        cloud_price=current_cloud_price,
                        cloud_queue_length=self._predicted_cloud_queue(
                            cloud_queue + 1
                        ),
                        cloud_platform_overhead_s=(
                            step_http_overhead_s
                            if step_http_overhead_s is not None
                            else self._predicted_platform_overhead_s(step)
                        ),
                        server_position=server_position,
                        price_history=tuple(self.price_history),
                        adjacency=adjacency,
                        v2v_throughput_by_link=throughput_by_link,
                        service_spatial_index=service_spatial_index,
                        v2v_path_index=v2v_path_index,
                        cloud_capacity_ratio=self._predicted_cloud_capacity_ratio(
                            cloud_queue + 1,
                            task.compute_cycles,
                        ),
                        cloud_target_offload_ratio=cloud_target_offload_ratio,
                    )
                    candidate_map = estimates(context, self.config)
                    all_actions_late = not any(
                        candidate.feasible
                        and math.isfinite(candidate.delay_s)
                        and candidate.delay_s <= task.deadline_s
                        for candidate in candidate_map.values()
                    )
                    v2v_target = candidate_map[OffloadAction.V2V].target_id
                    v2v_target_workload_s = (
                        vehicles[v2v_target].workload_cycles
                        / max(vehicles[v2v_target].compute_hz, 1.0)
                        if v2v_target in vehicles
                        else None
                    )
                    baseline_action = stackelberg_best_action(context, candidate_map, self.config)
                    oracle = min(
                        (
                            candidate
                            for candidate in candidate_map.values()
                            if candidate.feasible and math.isfinite(candidate.delay_s)
                        ),
                        key=lambda candidate: (candidate.delay_s, int(candidate.action)),
                    )
                    allowed_actions = policy_action_ids(
                        self.config.strategy, context, candidate_map, self.config
                    )
                    state = (
                        decision_state(context, candidate_map, self.config)
                        if learned_strategy
                        else None
                    )
                    phase_seconds["estimation"] += perf_counter() - phase_started
                    decisions.append({
                        "task": task,
                        "action": None,
                        "candidate_map": candidate_map,
                        "oracle": oracle,
                        "server_distance": server_distance,
                        "server_position": server_position,
                        "vehicle": vehicle,
                        "allowed_actions": allowed_actions,
                        "used_dqn": False,
                        "all_actions_late": all_actions_late,
                        "baseline_action": baseline_action,
                        "state": state,
                        "q_values": None,
                        "source_workload_s": source_workload_s,
                        "v2v_target_workload_s": v2v_target_workload_s,
                        "max_service_workload_s": snapshot_max_service_workload_s,
                        "cloud_queue_length": context.cloud_queue_length,
                        "step": step,
                        "estimate": None,
                        "backend_metadata": {},
                        "future": None,
                        "context": context,
                        "game_action": baseline_action,
                        "game_confidence": 0.0,
                        "hybrid_game_evidence": None,
                        "hybrid_dqn_evidence": None,
                        "hybrid_q_opposition": None,
                        "hybrid_cloud_pressure": None,
                        "hybrid_decision_source": "",
                        "anticipated_v2v_queue_s": 0.0,
                    })

                price_evaluation = None
                if (
                    self.config.cloud_pricing_mode == "follower_best_response"
                    and self.config.strategy in {"stackelberg", "hybrid_stackelberg"}
                    and decisions
                ):
                    phase_started = perf_counter()
                    (
                        current_cloud_price,
                        price_evaluation,
                        repriced,
                        quote_by_id,
                    ) = self._select_follower_responsive_price(
                        decisions,
                        vehicles,
                        current_cloud_price,
                        cloud_target_offload_ratio,
                        cloud_queue,
                    )
                    self._apply_repriced_decisions(decisions, repriced)
                    phase_seconds["topology_and_pricing"] += (
                        perf_counter() - phase_started
                    )
                pricing_record = {
                        "step": step,
                        "task_count": len(decisions),
                        "total_task_cycles": sum(
                            decision["task"].compute_cycles
                            for decision in decisions
                        ),
                        "mode": self.config.cloud_pricing_mode,
                        "strategy": self.config.strategy,
                        "selected_price": current_cloud_price,
                        "target_cloud_share": cloud_target_offload_ratio,
                        "predicted_cloud_share": (
                            price_evaluation.predicted_cloud_share
                            if price_evaluation is not None
                            else None
                        ),
                        "predicted_late_share": (
                            price_evaluation.predicted_late_share
                            if price_evaluation is not None
                            else None
                        ),
                        "predicted_cloud_requests": (
                            price_evaluation.predicted_cloud_requests
                            if price_evaluation is not None
                            else None
                        ),
                        "leader_score": (
                            price_evaluation.leader_score
                            if price_evaluation is not None
                            else None
                        ),
                        "leader_revenue_score": (
                            price_evaluation.revenue_score
                            if price_evaluation is not None
                            else None
                        ),
                        "capacity_violation": (
                            price_evaluation.capacity_violation
                            if price_evaluation is not None
                            else None
                        ),
                        "timeout_violation": (
                            price_evaluation.timeout_violation
                            if price_evaluation is not None
                            else None
                        ),
                        "candidate_count": (
                            self.config.cloud_price_candidate_count
                            if price_evaluation is not None
                            else 0
                        ),
                        "response_iterations": (
                            self.last_price_response_diagnostics.get("iterations")
                            if price_evaluation is not None
                            else None
                        ),
                        "response_request_residual": (
                            self.last_price_response_diagnostics.get(
                                "request_residual"
                            )
                            if price_evaluation is not None
                            else None
                        ),
                        "response_cycle_residual": (
                            self.last_price_response_diagnostics.get(
                                "cycle_residual"
                            )
                            if price_evaluation is not None
                            else None
                        ),
                        "outer_iterations": (
                            self.last_price_response_diagnostics.get(
                                "outer_iterations"
                            )
                            if price_evaluation is not None
                            else None
                        ),
                        "outer_request_residual": (
                            self.last_price_response_diagnostics.get(
                                "outer_request_residual"
                            )
                            if price_evaluation is not None
                            else None
                        ),
                        "outer_cycle_residual": (
                            self.last_price_response_diagnostics.get(
                                "outer_cycle_residual"
                            )
                            if price_evaluation is not None
                            else None
                        ),
                    }

                online_q_batch = None
                target_q_batch = None
                if learned_strategy and decisions:
                    phase_started = perf_counter()
                    states = [decision["state"] for decision in decisions]
                    online_q_batch = self.dqn.q_values_batch(states)
                    if (
                        self.config.strategy == "hybrid_stackelberg"
                        and self.config.decision.hybrid_fusion_mode == "residual"
                    ):
                        target_q_batch = self.dqn.q_values_batch(states, target=True)
                    phase_seconds["policy"] += perf_counter() - phase_started

                online_reliability = self._hybrid_online_reliability()
                arbitration_game_adequacy = self._arbitration_game_adequacy()
                for index, decision in enumerate(decisions):
                    phase_started = perf_counter()
                    online_q_values = (
                        online_q_batch[index] if online_q_batch is not None else None
                    )
                    target_q_values = (
                        target_q_batch[index] if target_q_batch is not None else None
                    )
                    action, state, used_dqn = choose_action(
                        self.config.strategy,
                        decision["context"],
                        decision["candidate_map"],
                        self.config,
                        self.policy_rng,
                        self.dqn,
                        explore=self.config.dqn.mode == "train",
                        precomputed_allowed_actions=decision["allowed_actions"],
                        precomputed_state=decision["state"],
                        online_q_values=online_q_values,
                        target_q_values=target_q_values,
                        advance_exploration=False,
                        online_reliability=online_reliability,
                        game_adequacy=arbitration_game_adequacy,
                    )
                    decision["action"] = action
                    decision["used_dqn"] = used_dqn
                    if self.config.strategy == "hybrid_stackelberg":
                        decision["hybrid_decision_source"] = (
                            hybrid_decision_source(
                                action,
                                used_dqn,
                                decision["candidate_map"],
                                decision["allowed_actions"],
                            )
                        )
                    guidance = game_guidance(
                        decision["context"],
                        decision["candidate_map"],
                        self.config,
                        decision["allowed_actions"],
                    )
                    decision["game_action"] = guidance.action
                    decision["game_confidence"] = guidance.confidence
                    arbitration = None
                    if (
                        self.config.strategy == "hybrid_stackelberg"
                        and self.config.decision.hybrid_fusion_mode
                        == "adaptive_confidence"
                    ):
                        assert online_q_values is not None
                        arbitration = hybrid_arbitration(
                            decision["context"],
                            guidance,
                            decision["allowed_actions"],
                            online_q_values,
                            self.config,
                            dqn_reliability=(
                                1.0
                                if self.config.dqn.mode == "evaluate"
                                else min(
                                    1.0,
                                    self.dqn.transition_count
                                    / max(
                                        self.config.dqn.warmup_transitions,
                                        1,
                                    ),
                                )
                            )
                            * online_reliability,
                            game_adequacy=arbitration_game_adequacy,
                        )
                        decision["hybrid_game_evidence"] = (
                            arbitration.game_evidence
                        )
                        decision["hybrid_dqn_evidence"] = (
                            arbitration.dqn_evidence
                        )
                        decision["hybrid_q_opposition"] = (
                            arbitration.q_opposition
                        )
                        decision["hybrid_cloud_pressure"] = (
                            arbitration.cloud_pressure
                        )
                    if arbitration is not None:
                        decision["game_preference_weight"] = (
                            min(1.0, arbitration.game_evidence)
                            if arbitration.use_game
                            else 0.0
                        )
                    else:
                        decision["game_preference_weight"] = (
                            min(
                                1.0,
                                guidance.confidence
                                / max(
                                    self.config.decision.hybrid_game_confidence_threshold,
                                    1e-12,
                                ),
                            )
                            if (
                                self.config.strategy == "hybrid_stackelberg"
                                and guidance.confidence
                                >= self.config.decision.hybrid_game_confidence_threshold
                            )
                            else 0.0
                        )
                    if self.config.record_decision_diagnostics:
                        decision["q_values"] = online_q_values
                    phase_seconds["policy"] += perf_counter() - phase_started

                    phase_started = perf_counter()
                    if self.config.dqn.mode == "train" and learned_strategy:
                        assert state is not None
                        task = decision["task"]
                        previous = pending_transitions.pop(task.vehicle_id, None)
                        if previous is not None:
                            (
                                previous_state,
                                previous_action,
                                previous_reward,
                                previous_game_action,
                                previous_game_confidence,
                                previous_allowed_actions,
                            ) = previous
                            transitions_to_store.append(
                                (
                                    previous_state,
                                    previous_action,
                                    previous_reward,
                                    state,
                                    False,
                                    decision["allowed_actions"],
                                    previous_game_action,
                                    previous_game_confidence,
                                    previous_allowed_actions,
                                )
                            )
                    phase_seconds["dqn_training"] += perf_counter() - phase_started

                self.decision_trace.apply(step, decisions)
                realized_cloud_cycles = sum(
                    decision["task"].compute_cycles
                    for decision in decisions
                    if decision["action"] == OffloadAction.V2I
                )
                total_step_cycles = sum(
                    decision["task"].compute_cycles
                    for decision in decisions
                )
                pricing_record["realized_cloud_share"] = (
                    realized_cloud_cycles / max(total_step_cycles, 1.0)
                )
                pricing_record["prediction_error"] = (
                    pricing_record["realized_cloud_share"]
                    - pricing_record["predicted_cloud_share"]
                    if pricing_record["predicted_cloud_share"] is not None
                    else None
                )
                recorder.record_pricing(pricing_record)

                if (
                    self.config.dqn.mode == "train"
                    and learned_strategy
                ):
                    self.dqn.advance_exploration(
                        True,
                        1 if self.config.dqn.epsilon_decay_per_step else len(tasks),
                    )
                    phase_started = perf_counter()
                    for transition in transitions_to_store:
                        self._store_transition(*transition)
                    phase_seconds["dqn_training"] += perf_counter() - phase_started

                v2i_this_step = 0
                max_service_workload_s = snapshot_max_service_workload_s
                pending_http_results: list[dict] = []
                for decision in sorted(
                    decisions,
                    key=lambda item: self._batch_arrival_key(step, item["task"].task_id),
                ):
                    phase_started = perf_counter()
                    task = decision["task"]
                    action = decision["action"]
                    vehicle = decision["vehicle"]
                    estimate = self._batch_adjusted_estimate(decision, vehicles)
                    actual_cloud_queue = self._predicted_cloud_queue(
                        cloud_queue + v2i_this_step + 1
                    )
                    if batch_http and action == OffloadAction.V2I:
                        future = self.backend.submit(task, actual_cloud_queue, step)
                        backend_metadata = {}
                    else:
                        future = None
                        estimate, backend_metadata = self._execute(
                            action,
                            estimate,
                            task,
                            actual_cloud_queue,
                            step,
                            vehicle,
                            decision["server_position"],
                        )
                    self._apply_resource_effects(
                        action,
                        estimate,
                        task,
                        vehicle,
                        vehicles,
                        quote_by_id,
                        current_cloud_price,
                    )
                    if action == OffloadAction.V2V and estimate.target_id in vehicles:
                        target = vehicles[estimate.target_id]
                        max_service_workload_s = max(
                            max_service_workload_s,
                            target.workload_cycles / max(target.compute_hz, 1.0),
                        )
                    if action == OffloadAction.V2I:
                        v2i_this_step += 1
                    decision["estimate"] = estimate
                    decision["backend_metadata"] = backend_metadata
                    decision["future"] = future
                    decision["max_service_workload_s"] = max_service_workload_s
                    phase_seconds["execution_and_state"] += perf_counter() - phase_started
                    if batch_http:
                        pending_http_results.append(decision)
                        continue
                    reward, success = reward_for(
                        estimate,
                        task,
                        self.config,
                        decision["context"],
                    )
                    phase_started = perf_counter()
                    if (
                        self.config.dqn.mode == "train"
                        and self.config.strategy in {"dqn", "hybrid_stackelberg"}
                    ):
                        state = decision["state"]
                        assert state is not None
                        pending_transitions[task.vehicle_id] = (
                            state,
                            int(action),
                            reward,
                            int(decision["game_action"]),
                            decision["game_preference_weight"],
                            tuple(decision["allowed_actions"]),
                        )
                    phase_seconds["dqn_training"] += perf_counter() - phase_started
                    phase_started = perf_counter()
                    self._record_override_outcome(decision, success)
                    result = self._make_result(
                        decision,
                        estimate,
                        backend_metadata,
                        reward,
                        success,
                    )
                    accumulator.add(result)
                    recorder.record(result)
                    phase_seconds["metrics_and_logging"] += perf_counter() - phase_started

                if batch_http:
                    for decision in pending_http_results:
                        phase_started = perf_counter()
                        estimate = decision["estimate"]
                        backend_metadata = decision["backend_metadata"]
                        future = decision["future"]
                        if future is not None:
                            measurement = future.result()
                            estimate, backend_metadata = self._apply_serverless_measurement(
                                estimate,
                                decision["task"],
                                decision["vehicle"],
                                decision["server_position"],
                                measurement,
                            )
                        reward, success = reward_for(
                            estimate,
                            decision["task"],
                            self.config,
                            decision["context"],
                        )
                        phase_seconds["execution_and_state"] += perf_counter() - phase_started
                        phase_started = perf_counter()
                        if (
                            self.config.dqn.mode == "train"
                            and self.config.strategy in {"dqn", "hybrid_stackelberg"}
                        ):
                            state = decision["state"]
                            assert state is not None
                            pending_transitions[decision["task"].vehicle_id] = (
                                state,
                                int(decision["action"]),
                                reward,
                                int(decision["game_action"]),
                                decision["game_preference_weight"],
                                tuple(decision["allowed_actions"]),
                            )
                        phase_seconds["dqn_training"] += perf_counter() - phase_started
                        phase_started = perf_counter()
                        self._record_override_outcome(decision, success)
                        result = self._make_result(
                            decision, estimate, backend_metadata, reward, success
                        )
                        accumulator.add(result)
                        recorder.record(result)
                        phase_seconds["metrics_and_logging"] += perf_counter() - phase_started
                cloud_queue = self._cloud_backlog_tasks()
                self.last_cloud_offload_ratio = v2i_this_step / max(len(tasks), 1)
                self.current_cloud_price = current_cloud_price
                self.price_history.append(current_cloud_price)
                self.price_history = self.price_history[-10:]
                segment = segments[step // segment_size]
                segment["wall_s"] += perf_counter() - step_started
                segment["tasks"] += len(tasks)
                segment["peak_active_vehicles"] = max(
                    segment["peak_active_vehicles"], len(vehicles)
                )
            if (
                self.config.dqn.mode == "train"
                and self.config.strategy in {"dqn", "hybrid_stackelberg"}
            ):
                for (
                    state,
                    action,
                    reward,
                    game_action,
                    game_confidence,
                    allowed_actions,
                ) in pending_transitions.values():
                    self._store_transition(
                        state,
                        action,
                        reward,
                        state,
                        True,
                        range(self.config.dqn.action_size),
                        game_action,
                        game_confidence,
                        allowed_actions,
                    )
            completed_trace = True
        finally:
            recorder.close()
            if mobility_started:
                self.mobility.close()
            if isinstance(self.backend, HttpKnativeBackend):
                self.backend.close()
            self.decision_trace.close(completed_trace)
        realized = len(departed) if departed else len(self.vehicle_energy)
        summary = accumulator.summary(
            self.config,
            completed_steps,
            realized,
            peak_active,
            len(self.dqn.replay),
            self.dqn.transition_count,
            self.dqn.update_count,
            self.dqn.epsilon,
        )
        run_dir = recorder.finish(summary)
        recorder.write_json(
            "decision-diagnostics.json",
            accumulator.diagnostics(completed_steps),
        )
        if (
            self.config.dqn.mode == "train"
            and self.config.strategy in {"dqn", "hybrid_stackelberg"}
        ):
            self.dqn.save(Path(run_dir) / "dqn-policy.pt")
        recorder.write_json(
            "timing.json",
            {
                "wall_clock_s": perf_counter() - run_started,
                "phase_seconds": phase_seconds,
                "segments": segments,
            },
        )
        return summary, str(run_dir)

    def _vehicle_states(self, frame_vehicles) -> dict[str, VehicleState]:
        return {
            item.vehicle_id: VehicleState(
                vehicle_id=item.vehicle_id,
                position=item.position,
                speed_mps=item.speed_mps,
                compute_hz=self.config.vehicle_compute_hz,
                energy_level=self.vehicle_energy.setdefault(item.vehicle_id, 1.0),
                queue_length=math.ceil(
                    self.service_workload_cycles.get(item.vehicle_id, 0.0)
                    / self.config.vehicle_compute_hz
                ),
                workload_cycles=self.service_workload_cycles.get(item.vehicle_id, 0.0),
            )
            for item in frame_vehicles
        }

    def _generate_tasks(self, step: int, vehicles: dict[str, VehicleState]) -> list[Task]:
        tasks: list[Task] = []
        for vehicle_id in sorted(vehicles):
            if vehicles[vehicle_id].is_service:
                continue
            if self.environment_rng.random() >= self.config.task_probability:
                continue
            if self.config.task_deadline_distribution == "discrete":
                deadline = self.environment_rng.choice(self.config.task_deadlines_s)
                deadline_min = min(self.config.task_deadlines_s)
                deadline_max = max(self.config.task_deadlines_s)
            else:
                deadline = self.environment_rng.uniform(
                    self.config.task_deadline_min_s,
                    self.config.task_deadline_max_s,
                )
                deadline_min = self.config.task_deadline_min_s
                deadline_max = self.config.task_deadline_max_s
            deadline_range = deadline_max - deadline_min
            urgency = (
                (deadline_max - deadline) / deadline_range
                if deadline_range > 0
                else 0.0
            )
            tasks.append(
                Task(
                    task_id=f"step-{step:05d}-{vehicle_id}",
                    vehicle_id=vehicle_id,
                    compute_cycles=self._sample_compute_cycles(),
                    data_size_mb=self.environment_rng.uniform(
                        self.config.task_data_min_mb, self.config.task_data_max_mb
                    ),
                    deadline_s=deadline,
                    urgency=max(0.0, min(1.0, urgency)),
                    created_step=step,
                )
            )
        return tasks

    def _batch_arrival_key(self, step: int, task_id: str) -> bytes:
        """Return a strategy-independent within-step arrival order."""
        return hashlib.blake2b(
            f"{self.config.seed}:{step}:{task_id}:arrival".encode("utf-8"),
            digest_size=8,
        ).digest()

    @staticmethod
    def _batch_adjusted_estimate(
        decision: dict,
        vehicles: dict[str, VehicleState],
    ) -> OffloadEstimate:
        """Replace expected same-batch V2V queueing with the realized queue."""
        estimate = decision["candidate_map"][decision["action"]]
        if estimate.action != OffloadAction.V2V or estimate.target_id not in vehicles:
            return estimate
        target = vehicles[estimate.target_id]
        current_workload_s = target.workload_cycles / max(target.compute_hz, 1.0)
        snapshot_workload_s = decision["v2v_target_workload_s"] or 0.0
        realized_queue_s = max(0.0, current_workload_s - snapshot_workload_s)
        expected_queue_s = max(
            0.0,
            float(decision.get("anticipated_v2v_queue_s", 0.0)),
        )
        correction_s = realized_queue_s - expected_queue_s
        if abs(correction_s) <= 1e-15:
            return estimate
        return replace(estimate, delay_s=estimate.delay_s + correction_s)

    def _execute(
        self,
        action: OffloadAction,
        estimate: OffloadEstimate,
        task: Task,
        queue_length: int,
        step: int,
        vehicle: VehicleState,
        server_position: tuple[float, float],
    ) -> tuple[OffloadEstimate, dict]:
        if action != OffloadAction.V2I:
            return estimate, {}
        measurement = self.backend.execute(task, queue_length, step)
        return self._apply_serverless_measurement(
            estimate, task, vehicle, server_position, measurement
        )

    def _apply_serverless_measurement(
        self,
        estimate: OffloadEstimate,
        task: Task,
        vehicle: VehicleState,
        server_position: tuple[float, float],
        measurement,
    ) -> tuple[OffloadEstimate, dict]:
        transmitted_task, preprocessing_delay, _ = offload_preprocessing(task, vehicle, self.config)
        radio_delay = transmission_delay_s(
            transmitted_task.data_size_mb,
            distance_m(vehicle.position, server_position),
            OffloadAction.V2I,
            self.config.network,
        )
        actual = replace(
            estimate,
            delay_s=preprocessing_delay + radio_delay + measurement.service_delay_s,
        )
        metadata = {
            "dispatch_queue_ms": measurement.dispatch_queue_ms,
            "http_latency_ms": measurement.http_latency_ms,
            "client_latency_ms": measurement.client_latency_ms,
            "processing_ms": measurement.processing_ms,
            "platform_overhead_ms": measurement.platform_overhead_ms,
            "preprocessing_delay_ms": preprocessing_delay * 1_000.0,
            "radio_delay_ms": radio_delay * 1_000.0,
            "physical_compute_ms": measurement.physical_compute_ms,
            "physical_queue_ms": measurement.physical_queue_ms,
            "scaled_processing_ms": measurement.scaled_processing_ms,
            "total_delay_ms": actual.delay_s * 1_000.0,
            "http_attempts": measurement.http_attempts,
            "http_retry_count": measurement.http_retry_count,
            "retry_backoff_ms": measurement.retry_backoff_ms,
            "cold_start": measurement.cold_start,
            "instance_id": measurement.instance_id,
            "checksum": measurement.checksum,
        }
        return actual, metadata

    def _apply_resource_effects(
        self,
        action: OffloadAction,
        estimate: OffloadEstimate,
        task: Task,
        vehicle: VehicleState,
        vehicles: dict[str, VehicleState],
        quote_by_id,
        current_cloud_price: float,
    ) -> None:
        remote_compute_energy = 0.0
        if action == OffloadAction.V2I:
            self.cloud_workload_cycles += task.compute_cycles
            remote_compute_energy = (
                task.compute_cycles
                / self.config.cloud_compute_hz
                * self.config.energy.cloud_compute_power_w
            )
        elif action == OffloadAction.V2V and estimate.target_id in vehicles:
            remote_compute_energy = (
                task.compute_cycles
                / vehicles[estimate.target_id].compute_hz
                * self.config.energy.service_compute_power_w
            )
        source_energy = max(0.0, estimate.energy_j - remote_compute_energy)
        vehicle.energy_level = max(
            0.0,
            vehicle.energy_level - source_energy / self.config.vehicle_battery_capacity_j,
        )
        self.vehicle_energy[vehicle.vehicle_id] = vehicle.energy_level
        if action == OffloadAction.LOCAL:
            vehicle.workload_cycles += task.compute_cycles
            self.service_workload_cycles[vehicle.vehicle_id] = vehicle.workload_cycles
            self.workload_compute_hz[vehicle.vehicle_id] = vehicle.compute_hz
            return
        if action != OffloadAction.V2V or estimate.target_id not in vehicles:
            return
        target = vehicles[estimate.target_id]
        target.workload_cycles += task.compute_cycles
        self.service_workload_cycles[target.vehicle_id] = target.workload_cycles
        self.workload_compute_hz[target.vehicle_id] = target.compute_hz
        target.queue_length = math.ceil(target.workload_cycles / max(target.compute_hz, 1.0))
        target.energy_level = max(
            0.0,
            target.energy_level
            - remote_compute_energy / self.config.service_vehicle_battery_capacity_j,
        )
        self.vehicle_energy[target.vehicle_id] = target.energy_level
        updated_quote = service_quote(
            target,
            current_cloud_price,
            self.config.service_price_sensitivity,
            self.config.service_min_energy,
            self.config.service_max_queue,
        )
        if updated_quote is None:
            quote_by_id.pop(target.vehicle_id, None)
        else:
            quote_by_id[target.vehicle_id] = updated_quote

    def _make_result(
        self,
        decision: dict,
        estimate: OffloadEstimate,
        backend_metadata: dict,
        reward: float,
        success: bool,
    ) -> OffloadResult:
        task = decision["task"]
        action = decision["action"]
        candidate_map = decision["candidate_map"]
        oracle = decision["oracle"]
        baseline_action = decision["baseline_action"]
        baseline_reward, _ = reward_for(
            candidate_map[baseline_action], task, self.config, decision["context"]
        )
        hybrid_deviation = (
            self.config.strategy == "hybrid_stackelberg" and action != baseline_action
        )
        q_values = decision["q_values"]
        allowed_actions = tuple(int(item) for item in decision["allowed_actions"])
        dqn_action = None
        dqn_q_margin = None
        if q_values is not None and allowed_actions:
            ranked_q = sorted(
                (
                    (float(q_values[action_id]), action_id)
                    for action_id in allowed_actions
                ),
                key=lambda item: (-item[0], item[1]),
            )
            dqn_action = OffloadAction(ranked_q[0][1])
            dqn_q_margin = (
                ranked_q[0][0] - ranked_q[1][0]
                if len(ranked_q) > 1
                else 0.0
            )
        return OffloadResult(
            task_id=task.task_id,
            vehicle_id=task.vehicle_id,
            action=action,
            delay_s=estimate.delay_s,
            energy_j=estimate.energy_j,
            payment=estimate.payment,
            reward=reward,
            success=success,
            step=decision["step"],
            target_id=estimate.target_id,
            path=estimate.path,
            dispatch_queue_ms=backend_metadata.get("dispatch_queue_ms"),
            http_latency_ms=backend_metadata.get("http_latency_ms"),
            client_latency_ms=backend_metadata.get("client_latency_ms"),
            processing_ms=backend_metadata.get("processing_ms"),
            platform_overhead_ms=backend_metadata.get("platform_overhead_ms"),
            preprocessing_delay_ms=backend_metadata.get("preprocessing_delay_ms"),
            radio_delay_ms=backend_metadata.get("radio_delay_ms"),
            physical_compute_ms=backend_metadata.get("physical_compute_ms"),
            physical_queue_ms=backend_metadata.get("physical_queue_ms"),
            scaled_processing_ms=backend_metadata.get("scaled_processing_ms"),
            total_delay_ms=backend_metadata.get("total_delay_ms"),
            http_attempts=backend_metadata.get("http_attempts"),
            http_retry_count=backend_metadata.get("http_retry_count"),
            retry_backoff_ms=backend_metadata.get("retry_backoff_ms"),
            cold_start=backend_metadata.get("cold_start"),
            instance_id=backend_metadata.get("instance_id"),
            checksum=backend_metadata.get("checksum"),
            server_distance_m=decision["server_distance"],
            oracle_action=oracle.action,
            oracle_delay_s=oracle.delay_s,
            decision_regret_s=max(0.0, estimate.delay_s - oracle.delay_s),
            oracle_success=oracle.delay_s <= task.deadline_s,
            task_compute_cycles=task.compute_cycles,
            task_data_size_mb=task.data_size_mb,
            effective_offload_data_mb=(
                task.data_size_mb * self.config.offload_compression_ratio
            ),
            task_deadline_s=task.deadline_s,
            local_estimate_s=candidate_map[OffloadAction.LOCAL].delay_s,
            v2v_estimate_s=candidate_map[OffloadAction.V2V].delay_s,
            v2i_estimate_s=candidate_map[OffloadAction.V2I].delay_s,
            allowed_action_count=len(decision["allowed_actions"]),
            used_dqn=decision["used_dqn"],
            stackelberg_action=baseline_action,
            hybrid_deviation=hybrid_deviation,
            hybrid_deviation_beneficial=(
                reward > baseline_reward if hybrid_deviation else None
            ),
            all_actions_late=decision["all_actions_late"],
            dqn_deviation=(hybrid_deviation and decision["used_dqn"]),
            rule_deviation=(hybrid_deviation and not decision["used_dqn"]),
            source_workload_s=decision["source_workload_s"],
            v2v_target_workload_s=decision["v2v_target_workload_s"],
            max_service_workload_s=decision["max_service_workload_s"],
            cloud_queue_length=decision["cloud_queue_length"],
            predicted_cloud_capacity_ratio=decision["context"].cloud_capacity_ratio,
            cloud_target_offload_ratio=(
                decision["context"].cloud_target_offload_ratio
            ),
            q_local=float(q_values[0]) if q_values is not None else None,
            q_v2v=float(q_values[1]) if q_values is not None else None,
            q_v2i=float(q_values[2]) if q_values is not None else None,
            dqn_action=dqn_action,
            dqn_q_margin=dqn_q_margin,
            cloud_price=decision["context"].cloud_price,
            game_action=decision["game_action"],
            game_confidence=decision["game_confidence"],
            hybrid_game_evidence=decision["hybrid_game_evidence"],
            hybrid_dqn_evidence=decision["hybrid_dqn_evidence"],
            hybrid_q_opposition=decision["hybrid_q_opposition"],
            hybrid_cloud_pressure=decision["hybrid_cloud_pressure"],
            hybrid_decision_source=decision["hybrid_decision_source"],
            metadata=backend_metadata,
        )

    def _select_follower_responsive_price(
        self,
        decisions: list[dict],
        vehicles: dict[str, VehicleState],
        anchor_price: float,
        target_cloud_share: float,
        current_cloud_requests: int,
    ) -> tuple[
        float,
        LeaderPriceEvaluation,
        list[dict],
        dict,
    ]:
        """Solve the cloud leader's bounded response to autonomous followers."""
        candidates = cloud_price_candidates(
            self.config.cloud_min_price,
            self.config.cloud_max_price,
            self.config.cloud_price_candidate_count,
            (
                self.config.cloud_base_price,
                self.current_cloud_price,
                anchor_price,
            ),
        )
        response_snapshot = self._price_response_snapshot(
            decisions,
            vehicles,
            current_cloud_requests,
        )
        total_cycles = sum(item["task"].compute_cycles for item in decisions)
        task_count = len(decisions)
        if self.config.cloud_price_batch_candidates:
            candidate_responses = self._solve_batch_follower_responses(
                response_snapshot,
                candidates,
                [target_cloud_share * task_count] * len(candidates),
                [target_cloud_share * total_cycles] * len(candidates),
            )
        else:
            candidate_responses = [
                self._solve_batch_follower_response(
                    response_snapshot,
                    price,
                    target_cloud_share * task_count,
                    target_cloud_share * total_cycles,
                )
                for price in candidates
            ]
        evaluated: list[LeaderPriceEvaluation] = []
        for price, response in zip(candidates, candidate_responses, strict=True):
            evaluation = evaluate_cloud_leader_response(
                price,
                response.cloud_cycles,
                total_cycles,
                response.late_tasks,
                task_count,
                target_cloud_share,
                self.config.cloud_base_price,
                self.config.cloud_max_price,
                self.config.cloud_capacity_price_weight,
                self.config.cloud_leader_timeout_weight,
                predicted_cloud_requests=response.cloud_requests,
                late_tolerance=self.config.cloud_leader_late_tolerance,
            )
            evaluated.append(evaluation)
        raw_evaluation = max(
            evaluated,
            key=lambda item: (
                item.leader_score,
                -abs(item.price - self.current_cloud_price),
                -item.price,
            ),
        )
        selected_price = (
            (1.0 - self.config.cloud_price_smoothing) * self.current_cloud_price
            + self.config.cloud_price_smoothing * raw_evaluation.price
        )
        selected_response = self._solve_batch_follower_response(
            response_snapshot,
            selected_price,
            raw_evaluation.predicted_cloud_requests,
            raw_evaluation.predicted_cloud_cycles,
        )
        final_response = selected_response
        final_snapshot = response_snapshot
        outer_iterations = 1
        outer_request_residual = 0.0
        outer_cycle_residual = 0.0
        for outer_index in range(1, self.config.cloud_price_outer_iterations):
            anticipated_v2v_queue_s = (
                self._expected_v2v_batch_queue_s(
                    final_snapshot,
                    final_response.action_probabilities,
                )
                if self.config.decision.synchronous_v2v_queue_forecast
                else None
            )
            repriced, _ = self._reprice_decisions(
                decisions,
                vehicles,
                selected_price,
                current_cloud_requests,
                final_response.cloud_requests,
                final_response.cloud_cycles,
                anticipated_v2v_queue_s,
            )
            next_snapshot = self._price_response_snapshot(
                repriced,
                vehicles,
                current_cloud_requests,
            )
            next_response = self._solve_batch_follower_response(
                next_snapshot,
                selected_price,
                final_response.cloud_requests,
                final_response.cloud_cycles,
            )
            outer_request_residual = abs(
                next_response.cloud_requests - final_response.cloud_requests
            ) / max(task_count, 1)
            outer_cycle_residual = abs(
                next_response.cloud_cycles - final_response.cloud_cycles
            ) / max(total_cycles, 1.0)
            final_response = next_response
            final_snapshot = next_snapshot
            outer_iterations = outer_index + 1
            tolerance = self.config.cloud_price_outer_tolerance
            if (
                tolerance > 0.0
                and outer_iterations >= self.config.cloud_price_outer_min_iterations
                and max(outer_request_residual, outer_cycle_residual) <= tolerance
            ):
                break
        anticipated_v2v_queue_s = (
            self._expected_v2v_batch_queue_s(
                final_snapshot,
                final_response.action_probabilities,
            )
            if self.config.decision.synchronous_v2v_queue_forecast
            else None
        )
        repriced, quotes = self._reprice_decisions(
            decisions,
            vehicles,
            selected_price,
            current_cloud_requests,
            final_response.cloud_requests,
            final_response.cloud_cycles,
            anticipated_v2v_queue_s,
        )
        selected_evaluation = evaluate_cloud_leader_response(
            selected_price,
            final_response.cloud_cycles,
            total_cycles,
            final_response.late_tasks,
            task_count,
            target_cloud_share,
            self.config.cloud_base_price,
            self.config.cloud_max_price,
            self.config.cloud_capacity_price_weight,
            self.config.cloud_leader_timeout_weight,
            predicted_cloud_requests=final_response.cloud_requests,
            late_tolerance=self.config.cloud_leader_late_tolerance,
        )
        self.last_price_response_diagnostics = {
            "iterations": final_response.iterations,
            "request_residual": final_response.request_residual,
            "cycle_residual": final_response.cycle_residual,
            "outer_iterations": outer_iterations,
            "outer_request_residual": outer_request_residual,
            "outer_cycle_residual": outer_cycle_residual,
        }
        return selected_price, selected_evaluation, repriced, quotes

    def _solve_batch_follower_responses(
        self,
        snapshot: dict,
        prices: list[float],
        initial_cloud_requests: list[float],
        initial_cloud_cycles: list[float],
    ) -> list[_BatchFollowerResponse]:
        """Solve candidate prices together so each iteration uses one Q batch."""
        if not (
            len(prices)
            == len(initial_cloud_requests)
            == len(initial_cloud_cycles)
        ):
            raise ValueError("candidate response vectors must have equal lengths")
        if not prices:
            return []
        anticipated_requests = np.maximum(
            np.asarray(initial_cloud_requests, dtype=np.float64),
            0.0,
        )
        anticipated_cycles = np.maximum(
            np.asarray(initial_cloud_cycles, dtype=np.float64),
            0.0,
        )
        relaxation = self.config.cloud_price_response_relaxation
        task_count = max(len(snapshot["cycles"]), 1)
        total_cycles = max(float(np.sum(snapshot["cycles"])), 1.0)
        responses: list[_BatchFollowerResponse] = []
        for iteration in range(self.config.cloud_price_response_iterations):
            responses = self._anticipated_cloud_demands_at_prices(
                snapshot,
                prices,
                anticipated_requests,
                anticipated_cycles,
            )
            for index, response in enumerate(responses):
                response.iterations = iteration + 1
                response.request_residual = abs(
                    response.cloud_requests - anticipated_requests[index]
                ) / task_count
                response.cycle_residual = abs(
                    response.cloud_cycles - anticipated_cycles[index]
                ) / total_cycles
            if iteration < self.config.cloud_price_response_iterations - 1:
                response_requests = np.asarray(
                    [response.cloud_requests for response in responses],
                    dtype=np.float64,
                )
                response_cycles = np.asarray(
                    [response.cloud_cycles for response in responses],
                    dtype=np.float64,
                )
                anticipated_requests = (
                    (1.0 - relaxation) * anticipated_requests
                    + relaxation * response_requests
                )
                anticipated_cycles = (
                    (1.0 - relaxation) * anticipated_cycles
                    + relaxation * response_cycles
                )
        return responses

    def _solve_batch_follower_response(
        self,
        snapshot: dict,
        price: float,
        initial_cloud_requests: float,
        initial_cloud_cycles: float,
    ) -> _BatchFollowerResponse:
        """Solve the symmetric batch response by relaxed fixed-point updates."""
        anticipated_requests = max(initial_cloud_requests, 0.0)
        anticipated_cycles = max(initial_cloud_cycles, 0.0)
        relaxation = self.config.cloud_price_response_relaxation
        response = _BatchFollowerResponse(
            cloud_cycles=0.0,
            late_tasks=0.0,
            cloud_requests=0.0,
            action_probabilities=np.empty((0, self.config.dqn.action_size)),
        )
        for iteration in range(self.config.cloud_price_response_iterations):
            response = self._anticipated_cloud_demand_at_price(
                snapshot,
                price,
                anticipated_requests,
                anticipated_cycles,
            )
            request_residual = abs(
                response.cloud_requests - anticipated_requests
            ) / max(len(snapshot["cycles"]), 1)
            cycle_residual = abs(
                response.cloud_cycles - anticipated_cycles
            ) / max(float(np.sum(snapshot["cycles"])), 1.0)
            response.iterations = iteration + 1
            response.request_residual = request_residual
            response.cycle_residual = cycle_residual
            tolerance = self.config.cloud_price_response_tolerance
            if (
                tolerance > 0.0
                and response.iterations
                >= self.config.cloud_price_response_min_iterations
                and max(request_residual, cycle_residual) <= tolerance
            ):
                break
            if iteration < self.config.cloud_price_response_iterations - 1:
                anticipated_requests = (
                    (1.0 - relaxation) * anticipated_requests
                    + relaxation * response.cloud_requests
                )
                anticipated_cycles = (
                    (1.0 - relaxation) * anticipated_cycles
                    + relaxation * response.cloud_cycles
                )
        return response

    def _price_response_states(
        self,
        snapshot: dict,
        price: float,
        anticipated_queue: int,
        anticipated_capacity_ratio: float,
        delays: np.ndarray,
        payments: np.ndarray,
    ) -> np.ndarray:
        """Build the same public state that followers receive after repricing."""
        states = snapshot["states"].copy()
        decision = self.config.decision
        states[:, 8] = np.minimum(
            np.maximum(price / max(self.config.cloud_max_price, 1e-12), 0.0),
            2.0,
        )
        states[:, 10] = np.minimum(max(anticipated_capacity_ratio, 0.0), 2.0)
        states[:, 17] = np.minimum(
            np.maximum(
                payments[:, int(OffloadAction.V2V)]
                / max(decision.payment_scale, 1e-12),
                0.0,
            ),
            2.0,
        )
        states[:, 18] = np.minimum(
            np.maximum(
                payments[:, int(OffloadAction.V2I)]
                / max(decision.payment_scale, 1e-12),
                0.0,
            ),
            2.0,
        )
        if self.config.cloud_price_state_consistency:
            states[:, 9] = np.minimum(
                max(
                    anticipated_queue
                    / max(self.config.network.queue_delay_threshold, 1),
                    0.0,
                ),
                2.0,
            )
            states[:, 11:14] = np.minimum(
                np.maximum(
                    delays / np.maximum(snapshot["deadlines"][:, None], 1e-12),
                    0.0,
                ),
                2.0,
            )
        return states.astype(np.float32, copy=False)

    @staticmethod
    def _expected_v2v_batch_queue_s(
        snapshot: dict,
        action_probabilities: np.ndarray,
    ) -> np.ndarray:
        """Expected prior V2V work under a strategy-independent random order."""
        cycles = np.asarray(snapshot["cycles"], dtype=np.float64)
        task_count = len(cycles)
        if action_probabilities.shape != (
            task_count,
            int(OffloadAction.V2I) + 1,
        ):
            raise ValueError("action probability shape does not match batch")
        probabilities = action_probabilities[:, int(OffloadAction.V2V)]
        expected_cycles = probabilities * cycles
        queue_s = np.zeros(task_count, dtype=np.float64)
        target_indices = snapshot["target_indices"]
        valid = target_indices >= 0
        if not np.any(valid):
            return queue_s
        total_by_target = np.bincount(
            target_indices[valid],
            weights=expected_cycles[valid],
        )
        other_cycles = np.maximum(
            0.0,
            total_by_target[target_indices[valid]] - expected_cycles[valid],
        )
        queue_s[valid] = (
            0.5
            * other_cycles
            / np.maximum(snapshot["target_compute_hz"][valid], 1.0)
        )
        return queue_s

    def _price_response_snapshot(
        self,
        decisions: list[dict],
        vehicles: dict[str, VehicleState],
        current_cloud_requests: int,
    ) -> dict:
        """Pack price-invariant follower data into contiguous arrays."""
        task_count = len(decisions)
        cycles = np.asarray(
            [decision["task"].compute_cycles for decision in decisions],
            dtype=np.float64,
        )
        deadlines = np.asarray(
            [decision["task"].deadline_s for decision in decisions],
            dtype=np.float64,
        )
        urgency = np.asarray(
            [decision["task"].urgency for decision in decisions],
            dtype=np.float64,
        )
        delays = np.asarray(
            [
                [
                    decision["candidate_map"][action].delay_s
                    for action in OffloadAction
                ]
                for decision in decisions
            ],
            dtype=np.float64,
        )
        energies = np.asarray(
            [
                [
                    decision["candidate_map"][action].energy_j
                    for action in OffloadAction
                ]
                for decision in decisions
            ],
            dtype=np.float64,
        )
        feasible = np.asarray(
            [
                [
                    decision["candidate_map"][action].feasible
                    and math.isfinite(
                        decision["candidate_map"][action].delay_s
                    )
                    for action in OffloadAction
                ]
                for decision in decisions
            ],
            dtype=np.bool_,
        )
        allowed = np.zeros((task_count, self.config.dqn.action_size), dtype=np.bool_)
        for index, decision in enumerate(decisions):
            actions = decision["allowed_actions"]
            if (
                self.config.strategy == "stackelberg"
                and self.config.decision.stackelberg_deadline_action_masking
            ):
                on_time = [
                    int(action)
                    for action in OffloadAction
                    if feasible[index, int(action)]
                    and delays[index, int(action)] <= deadlines[index]
                ]
                actions = on_time or actions
            allowed[index, actions] = True
        target_ids = [
            decision["candidate_map"][OffloadAction.V2V].target_id
            for decision in decisions
        ]
        target_counts: dict[str, int] = {}
        target_index_by_id: dict[str, int] = {}
        target_indices = np.full(task_count, -1, dtype=np.int32)
        target_compute_hz = np.ones(task_count, dtype=np.float64)
        for index, target_id in enumerate(target_ids):
            target = vehicles.get(target_id or "")
            if target_id is None or target is None:
                continue
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
            target_indices[index] = target_index_by_id.setdefault(
                target_id,
                len(target_index_by_id),
            )
            target_compute_hz[index] = target.compute_hz
        service_count = max(
            sum(vehicle.is_service for vehicle in vehicles.values()),
            1,
        )
        average_demand = task_count / service_count
        demand_ratios = {
            target_id: count / max(average_demand, 1.0)
            for target_id, count in target_counts.items()
        }
        return {
            "cycles": cycles,
            "deadlines": deadlines,
            "urgency": urgency,
            "delays": delays,
            "energies": energies,
            "feasible": feasible,
            "allowed": allowed,
            "target_ids": target_ids,
            "target_indices": target_indices,
            "target_compute_hz": target_compute_hz,
            "demand_ratios": demand_ratios,
            "vehicles": vehicles,
            "states": (
                np.stack([decision["state"] for decision in decisions])
                if self.config.strategy == "hybrid_stackelberg"
                else None
            ),
            "cloud_capacity_ratios": np.asarray(
                [
                    decision["context"].cloud_capacity_ratio
                    for decision in decisions
                ],
                dtype=np.float64,
            ),
            "current_cloud_requests": max(current_cloud_requests, 0),
            "base_cloud_queue_length": (
                decisions[0]["context"].cloud_queue_length
                if decisions
                else 0
            ),
        }

    def _anticipated_cloud_demand_at_price(
        self,
        snapshot: dict,
        price: float,
        anticipated_cloud_requests: float = 0.0,
        anticipated_cloud_cycles: float = 0.0,
    ) -> _BatchFollowerResponse:
        prepared = self._prepare_anticipated_cloud_demand(
            snapshot,
            price,
            anticipated_cloud_requests,
            anticipated_cloud_cycles,
        )
        q_values = (
            self.dqn.q_values_batch(prepared.states)
            if prepared.states is not None
            else None
        )
        return self._complete_anticipated_cloud_demand(prepared, q_values)

    def _anticipated_cloud_demands_at_prices(
        self,
        snapshot: dict,
        prices: list[float],
        anticipated_cloud_requests: np.ndarray,
        anticipated_cloud_cycles: np.ndarray,
    ) -> list[_BatchFollowerResponse]:
        prepared = [
            self._prepare_anticipated_cloud_demand(
                snapshot,
                price,
                float(requests),
                float(cycles),
            )
            for price, requests, cycles in zip(
                prices,
                anticipated_cloud_requests,
                anticipated_cloud_cycles,
                strict=True,
            )
        ]
        if self.config.strategy == "stackelberg":
            return [
                self._complete_anticipated_cloud_demand(item, None)
                for item in prepared
            ]
        states = [item.states for item in prepared]
        if any(item is None for item in states):
            raise AssertionError("Hybrid price response requires policy states")
        task_count = len(snapshot["cycles"])
        combined = np.concatenate(states, axis=0)
        combined_q = self.dqn.q_values_batch(combined)
        return [
            self._complete_anticipated_cloud_demand(
                item,
                combined_q[index * task_count : (index + 1) * task_count],
            )
            for index, item in enumerate(prepared)
        ]

    def _prepare_anticipated_cloud_demand(
        self,
        snapshot: dict,
        price: float,
        anticipated_cloud_requests: float,
        anticipated_cloud_cycles: float,
    ) -> _PreparedFollowerResponse:
        cycles = snapshot["cycles"]
        deadlines = snapshot["deadlines"]
        delays = snapshot["delays"].copy()
        energies = snapshot["energies"]
        feasible = snapshot["feasible"]
        task_count = len(cycles)
        expected_request_rank = (
            snapshot["current_cloud_requests"]
            + 0.5 * max(anticipated_cloud_requests, 0.0)
        )
        anticipated_queue = self._predicted_cloud_queue(
            math.ceil(expected_request_rank)
        )
        base_queue = snapshot["base_cloud_queue_length"]
        delays[:, int(OffloadAction.V2I)] += (
            cloud_queue_delay_s(anticipated_queue, self.config.network)
            - cloud_queue_delay_s(base_queue, self.config.network)
        )
        anticipated_capacity_ratio = self._batch_cloud_capacity_ratio(
            snapshot["current_cloud_requests"],
            anticipated_cloud_requests,
            anticipated_cloud_cycles,
        )
        allowed = feasible.copy()
        apply_deadline_mask = (
            (
                self.config.strategy == "hybrid_stackelberg"
                and self.config.decision.deadline_action_masking
            )
            or (
                self.config.strategy == "stackelberg"
                and self.config.decision.stackelberg_deadline_action_masking
            )
        )
        if apply_deadline_mask:
            on_time = feasible & (delays <= deadlines[:, None])
            any_on_time = np.any(on_time, axis=1)
            allowed = np.where(any_on_time[:, None], on_time, feasible)
        payments = np.zeros((task_count, self.config.dqn.action_size), dtype=np.float64)
        payments[:, int(OffloadAction.V2I)] = price * (cycles / 1e9)
        quote_prices = {}
        for target_id in set(snapshot["target_ids"]):
            if target_id is None:
                continue
            vehicle = snapshot["vehicles"].get(target_id)
            if vehicle is None:
                continue
            quote = service_quote(
                vehicle,
                price,
                self.config.service_price_sensitivity,
                self.config.service_min_energy,
                self.config.service_max_queue,
                anticipated_demand_ratio=snapshot["demand_ratios"].get(
                    target_id,
                    0.0,
                ),
                demand_price_weight=self.config.service_demand_price_weight,
            )
            if quote is not None:
                quote_prices[target_id] = quote.price
        payments[:, int(OffloadAction.V2V)] = np.asarray(
            [
                quote_prices.get(target_id, math.inf) * task_cycles / 1e9
                for target_id, task_cycles in zip(
                    snapshot["target_ids"],
                    cycles,
                )
            ],
            dtype=np.float64,
        )
        decision = self.config.decision
        objectives = (
            decision.delay_weight * delays / deadlines[:, None]
            + decision.energy_weight * energies / decision.energy_scale_j
            + decision.payment_weight * payments / decision.payment_scale
        )
        on_time_bonus = (
            decision.stackelberg_on_time_bonus
            * (1.0 + snapshot["urgency"][:, None])
            * (delays <= deadlines[:, None])
        )
        objectives = np.where(allowed, objectives - on_time_bonus, math.inf)
        order = np.argsort(objectives, axis=1, kind="stable")
        rows = np.arange(task_count)
        game_actions = order[:, 0]
        best_cost = objectives[rows, game_actions]
        second_cost = objectives[rows, order[:, 1]]
        confidence = np.maximum(0.0, second_cost - best_cost) / np.maximum(
            np.abs(best_cost),
            1.0,
        )
        metrics = np.stack((delays, energies, payments), axis=2)
        best_metrics = metrics[rows, game_actions]
        dominates = np.ones(task_count, dtype=np.bool_)
        for action in range(self.config.dqn.action_size):
            comparison_needed = allowed[:, action] & (game_actions != action)
            no_worse = np.all(best_metrics <= metrics[:, action, :], axis=1)
            strictly_better = np.any(best_metrics < metrics[:, action, :], axis=1)
            dominates &= ~comparison_needed | (no_worse & strictly_better)

        states = (
            self._price_response_states(
                snapshot,
                price,
                anticipated_queue,
                anticipated_capacity_ratio,
                delays,
                payments,
            )
            if self.config.strategy != "stackelberg"
            else None
        )
        return _PreparedFollowerResponse(
            cycles=cycles,
            deadlines=deadlines,
            delays=delays,
            allowed=allowed,
            game_actions=game_actions,
            confidence=confidence,
            dominates=dominates,
            anticipated_capacity_ratio=anticipated_capacity_ratio,
            states=states,
        )

    def _complete_anticipated_cloud_demand(
        self,
        prepared: _PreparedFollowerResponse,
        q_values: np.ndarray | None,
    ) -> _BatchFollowerResponse:
        cycles = prepared.cycles
        deadlines = prepared.deadlines
        delays = prepared.delays
        allowed = prepared.allowed
        game_actions = prepared.game_actions
        confidence = prepared.confidence
        dominates = prepared.dominates
        anticipated_capacity_ratio = prepared.anticipated_capacity_ratio
        task_count = len(cycles)
        rows = np.arange(task_count)
        game_probabilities = np.eye(
            self.config.dqn.action_size,
            dtype=np.float64,
        )[game_actions]
        if self.config.strategy == "stackelberg":
            action_probabilities = game_probabilities
        else:
            if q_values is None:
                raise AssertionError("Hybrid price response requires Q values")
            masked_q = np.where(allowed, q_values, -math.inf)
            if self.config.cloud_price_response_policy == "argmax":
                learned_actions = np.argmax(masked_q, axis=1)
                learned_probabilities = np.eye(
                    self.config.dqn.action_size,
                    dtype=np.float64,
                )[learned_actions]
            else:
                logits = masked_q / self.config.cloud_price_response_temperature
                logits -= np.max(logits, axis=1, keepdims=True)
                weights = np.exp(logits)
                learned_probabilities = (
                    weights
                    / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
                )
            reliability = (
                1.0
                if self.config.dqn.mode == "evaluate"
                else min(
                    1.0,
                    self.dqn.transition_count
                    / max(self.config.dqn.warmup_transitions, 1),
                )
            )
            action_probabilities = (
                reliability * learned_probabilities
                + (1.0 - reliability) * game_probabilities
            )
            if (
                self.config.decision.hybrid_fusion_mode
                == "adaptive_confidence"
            ):
                dqn_actions = np.argmax(masked_q, axis=1)
                best_q = masked_q[rows, dqn_actions]
                allowed_max = np.max(masked_q, axis=1)
                allowed_min = np.min(
                    np.where(allowed, q_values, math.inf),
                    axis=1,
                )
                q_range = np.maximum(allowed_max - allowed_min, 1e-12)
                q_opposition = np.maximum(
                    0.0,
                    best_q - q_values[rows, game_actions],
                ) / q_range
                game_evidence = confidence / max(
                    self.config.decision.hybrid_game_confidence_threshold,
                    1e-12,
                )
                dqn_evidence = (
                    reliability
                    * q_opposition
                    / self.config.decision.hybrid_dqn_opposition_threshold
                )
                capacity_target = (
                    self.config.serverless.capacity_utilization_target
                )
                cloud_pressure = np.maximum(
                    0.0,
                    anticipated_capacity_ratio
                    / max(capacity_target, 1e-12)
                    - 1.0,
                )
                externality = (
                    1.0
                    + self.config.decision.hybrid_congestion_sensitivity
                    * cloud_pressure
                )
                game_cloud = game_actions == int(OffloadAction.V2I)
                dqn_cloud = dqn_actions == int(OffloadAction.V2I)
                dqn_evidence = np.where(
                    game_cloud & ~dqn_cloud,
                    dqn_evidence * externality,
                    dqn_evidence,
                )
                game_evidence = np.where(
                    ~game_cloud & dqn_cloud,
                    game_evidence * externality,
                    game_evidence,
                )
                hard_response = (game_actions == dqn_actions) | (
                    game_evidence >= dqn_evidence
                )
                hard_response |= dominates
            elif self.config.decision.hybrid_fusion_mode == "delegated":
                hard_response = dominates
            else:
                hard_response = dominates | (
                    confidence
                    >= self.config.decision.hybrid_game_confidence_threshold
                )
            action_probabilities = np.where(
                hard_response[:, None],
                game_probabilities,
                action_probabilities,
            )
        cloud_probability = action_probabilities[:, int(OffloadAction.V2I)]
        cloud_cycles = float(np.dot(cloud_probability, cycles))
        cloud_requests = float(cloud_probability.sum())
        late_tasks = float(
            cloud_probability[
                delays[:, int(OffloadAction.V2I)] > deadlines
            ].sum()
        )
        return _BatchFollowerResponse(
            cloud_cycles=cloud_cycles,
            late_tasks=late_tasks,
            cloud_requests=cloud_requests,
            action_probabilities=action_probabilities,
        )

    def _reprice_decisions(
        self,
        decisions: list[dict],
        vehicles: dict[str, VehicleState],
        price: float,
        current_cloud_requests: int,
        anticipated_cloud_requests: float,
        anticipated_cloud_cycles: float,
        anticipated_v2v_queue_s: np.ndarray | None = None,
    ) -> tuple[list[dict], dict]:
        """Re-evaluate follower utilities at one leader price without acting."""
        target_demand: dict[str, int] = {}
        for decision in decisions:
            target_id = decision["candidate_map"][OffloadAction.V2V].target_id
            if target_id is not None:
                target_demand[target_id] = target_demand.get(target_id, 0) + 1
        average_demand = len(decisions) / max(
            sum(vehicle.is_service for vehicle in vehicles.values()),
            1,
        )
        quotes = {}
        for vehicle in vehicles.values():
            if not vehicle.is_service:
                continue
            quote = service_quote(
                vehicle,
                price,
                self.config.service_price_sensitivity,
                self.config.service_min_energy,
                self.config.service_max_queue,
                anticipated_demand_ratio=(
                    target_demand.get(vehicle.vehicle_id, 0)
                    / max(average_demand, 1.0)
                ),
                demand_price_weight=self.config.service_demand_price_weight,
            )
            if quote is not None:
                quotes[quote.vehicle_id] = quote
        repriced = []
        repriced_spatial_index = None
        learned = self.config.strategy == "hybrid_stackelberg"
        anticipated_queue = self._predicted_cloud_queue(
            math.ceil(
                max(current_cloud_requests, 0)
                + 0.5 * max(anticipated_cloud_requests, 0.0)
            )
        )
        anticipated_capacity_ratio = self._batch_cloud_capacity_ratio(
            current_cloud_requests,
            anticipated_cloud_requests,
            anticipated_cloud_cycles,
        )
        for index, decision in enumerate(decisions):
            task = decision["task"]
            context = replace(
                decision["context"],
                cloud_price=price,
                cloud_queue_length=anticipated_queue,
                cloud_capacity_ratio=anticipated_capacity_ratio,
                service_quotes=quotes,
            )
            candidate_map = dict(decision["candidate_map"])
            candidate_map[OffloadAction.V2I] = v2i_estimate(
                context,
                self.config,
            )
            v2v = candidate_map[OffloadAction.V2V]
            quote = quotes.get(v2v.target_id or "")
            if v2v.feasible and quote is not None:
                candidate_map[OffloadAction.V2V] = replace(
                    v2v,
                    payment=quote.price * (task.compute_cycles / 1e9),
                )
            elif v2v.feasible:
                if repriced_spatial_index is None:
                    repriced_spatial_index = ServiceSpatialIndex(vehicles, quotes)
                    repriced_spatial_index.prepare_sources(
                        {
                            candidate["task"].vehicle_id
                            for candidate in decisions
                        },
                        self.config.network.neighbor_radius_m,
                    )
                fallback_context = replace(
                    context,
                    v2v_path_index=None,
                    service_spatial_index=repriced_spatial_index,
                )
                candidate_map[OffloadAction.V2V] = estimates(
                    fallback_context,
                    self.config,
                )[OffloadAction.V2V]
            v2v_queue_s = (
                max(0.0, float(anticipated_v2v_queue_s[index]))
                if anticipated_v2v_queue_s is not None
                else 0.0
            )
            v2v = candidate_map[OffloadAction.V2V]
            if v2v.feasible and math.isfinite(v2v.delay_s) and v2v_queue_s > 0.0:
                candidate_map[OffloadAction.V2V] = replace(
                    v2v,
                    delay_s=v2v.delay_s + v2v_queue_s,
                )
            allowed = policy_action_ids(
                self.config.strategy,
                context,
                candidate_map,
                self.config,
            )
            if (
                self.config.strategy == "stackelberg"
                and self.config.decision.stackelberg_deadline_action_masking
            ):
                on_time = [
                    int(candidate.action)
                    for candidate in candidate_map.values()
                    if candidate.feasible
                    and math.isfinite(candidate.delay_s)
                    and candidate.delay_s <= task.deadline_s
                ]
                allowed = on_time or allowed
            guidance = game_guidance(
                context,
                candidate_map,
                self.config,
                allowed,
            )
            repriced.append(
                {
                    "context": context,
                    "candidate_map": candidate_map,
                    "allowed_actions": allowed,
                    "state": (
                        decision_state(context, candidate_map, self.config)
                        if learned
                        else None
                    ),
                    "guidance": guidance,
                    "task": task,
                    "anticipated_v2v_queue_s": v2v_queue_s,
                }
            )
        return repriced, quotes

    @staticmethod
    def _apply_repriced_decisions(
        decisions: list[dict],
        repriced: list[dict],
    ) -> None:
        for decision, updated in zip(decisions, repriced):
            candidate_map = updated["candidate_map"]
            finite = [
                candidate
                for candidate in candidate_map.values()
                if candidate.feasible and math.isfinite(candidate.delay_s)
            ]
            decision["context"] = updated["context"]
            decision["candidate_map"] = candidate_map
            decision["allowed_actions"] = updated["allowed_actions"]
            decision["state"] = updated["state"]
            decision["baseline_action"] = updated["guidance"].action
            decision["game_action"] = updated["guidance"].action
            decision["game_confidence"] = updated["guidance"].confidence
            decision["anticipated_v2v_queue_s"] = updated[
                "anticipated_v2v_queue_s"
            ]
            decision["all_actions_late"] = not any(
                candidate.delay_s <= decision["task"].deadline_s
                for candidate in finite
            )
            if finite:
                decision["oracle"] = min(
                    finite,
                    key=lambda candidate: (
                        candidate.delay_s,
                        int(candidate.action),
                    ),
                )

    def _create_backend(self) -> ServerlessBackend:
        if self.config.backend == "knative":
            return HttpKnativeBackend(
                self.config.serverless,
                cloud_compute_hz=self.config.cloud_compute_hz,
                queue_delay_fn=lambda queue: cloud_queue_delay_s(
                    queue, self.config.network
                ),
            )
        return AnalyticalServerlessBackend(
            cloud_compute_hz=self.config.cloud_compute_hz,
            cold_start_s=self.config.network.analytical_cold_start_s,
            queue_delay_fn=lambda queue: cloud_queue_delay_s(queue, self.config.network),
            idle_steps_to_zero=self.config.serverless.idle_steps_to_zero,
        )

    def _scaled_cloud_queue(self, request_count: int) -> int:
        request_count = max(0, request_count)
        if request_count == 0:
            return 0
        target = self.config.serverless.concurrency_target
        instances = min(self.config.serverless.max_instances, max(1, math.ceil(request_count / target)))
        return math.ceil(request_count / instances)

    def _cloud_capacity_target(self, tasks: list[Task]) -> float:
        """Return the largest offload share inside request and compute reserves."""
        if not tasks:
            return self.config.cloud_target_offload_ratio
        reserve = self.config.serverless.capacity_utilization_target
        request_capacity = (
            self.config.serverless.max_instances
            * self.config.serverless.concurrency_target
        )
        compute_capacity = (
            self.config.serverless.max_instances * self.config.cloud_compute_hz
        )
        request_share = reserve * request_capacity / len(tasks)
        compute_share = reserve * compute_capacity / sum(
            task.compute_cycles for task in tasks
        )
        return max(
            1e-6,
            min(
                self.config.cloud_target_offload_ratio,
                request_share,
                compute_share,
                1.0,
            ),
        )

    def _cloud_utilization_ratio(self) -> float:
        compute_capacity = (
            self.config.serverless.max_instances * self.config.cloud_compute_hz
        )
        return self.cloud_workload_cycles / max(compute_capacity, 1.0)

    def _predicted_cloud_capacity_ratio(
        self,
        request_count: int,
        candidate_cycles: float,
    ) -> float:
        request_capacity = (
            self.config.serverless.max_instances
            * self.config.serverless.concurrency_target
        )
        compute_capacity = (
            self.config.serverless.max_instances * self.config.cloud_compute_hz
        )
        request_ratio = max(0, request_count) / request_capacity
        compute_ratio = (self.cloud_workload_cycles + candidate_cycles) / compute_capacity
        return max(request_ratio, compute_ratio)

    def _batch_cloud_capacity_ratio(
        self,
        current_request_count: int,
        anticipated_request_count: float,
        anticipated_cycles: float,
    ) -> float:
        """Return full-batch load pressure for the leader's broadcast state."""
        request_capacity = (
            self.config.serverless.max_instances
            * self.config.serverless.concurrency_target
        )
        compute_capacity = (
            self.config.serverless.max_instances
            * self.config.cloud_compute_hz
        )
        request_ratio = (
            max(current_request_count, 0)
            + max(anticipated_request_count, 0.0)
        ) / request_capacity
        compute_ratio = (
            self.cloud_workload_cycles
            + max(anticipated_cycles, 0.0)
        ) / compute_capacity
        return max(request_ratio, compute_ratio)

    def _predicted_cloud_queue(self, request_count: int) -> int:
        return self._scaled_cloud_queue(request_count)

    def _predicted_platform_overhead_s(self, step: int) -> float:
        if isinstance(self.backend, AnalyticalServerlessBackend):
            return self.config.network.analytical_cold_start_s if self.backend.will_cold_start(step) else 0.0
        return self.backend.predicted_platform_overhead_s()

    def _sample_compute_cycles(self) -> float:
        if self.config.task_compute_distribution == "discrete":
            return self.environment_rng.choice(self.config.task_compute_choices)
        return self.environment_rng.uniform(
            self.config.task_compute_min_cycles,
            self.config.task_compute_max_cycles,
        )

    def _is_fixed_service_vehicle(self, vehicle_id: str) -> bool:
        digest = hashlib.blake2b(
            f"{self.config.seed}:{vehicle_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        unit_interval = int.from_bytes(digest, "big") / float(1 << 64)
        return unit_interval < self.config.service_vehicle_ratio

    def _advance_workloads(self) -> None:
        for vehicle_id, workload in tuple(self.service_workload_cycles.items()):
            capacity = self.workload_compute_hz.get(vehicle_id, self.config.vehicle_compute_hz)
            remaining = max(0.0, workload - capacity)
            if remaining:
                self.service_workload_cycles[vehicle_id] = remaining
            else:
                self.service_workload_cycles.pop(vehicle_id, None)
                self.workload_compute_hz.pop(vehicle_id, None)
        backlog = self._cloud_backlog_tasks()
        instances = min(
            self.config.serverless.max_instances,
            max(1, math.ceil(backlog / self.config.serverless.concurrency_target)),
        )
        self.cloud_workload_cycles = max(
            0.0,
            self.cloud_workload_cycles - self.config.cloud_compute_hz * instances,
        )

    def _cloud_backlog_tasks(self) -> int:
        if self.cloud_workload_cycles <= 0:
            return 0
        mean_cycles = (
            sum(self.config.task_compute_choices) / len(self.config.task_compute_choices)
            if self.config.task_compute_distribution == "discrete"
            else (self.config.task_compute_min_cycles + self.config.task_compute_max_cycles) / 2.0
        )
        return math.ceil(self.cloud_workload_cycles / mean_cycles)

    def _store_transition(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        next_allowed_actions,
        preferred_action: int = -1,
        preference_weight: float = 0.0,
        preferred_allowed_actions=None,
    ) -> None:
        self.dqn.store(
            state,
            action,
            reward,
            next_state,
            done,
            next_allowed_actions,
            preferred_action,
            preference_weight,
            preferred_allowed_actions,
        )
        if self.dqn.transition_count % self.config.dqn.training_interval == 0:
            self.dqn.update()
