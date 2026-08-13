from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from collections.abc import Mapping

import numpy as np

from .config import SimulationConfig
from .domain import OffloadAction, OffloadEstimate, ServiceQuote, Task, VehicleState
from .dqn import DQNAgent, build_state
from .network import (
    ReverseParetoV2VIndex,
    ServiceSpatialIndex,
    best_v2v_path,
    cloud_queue_delay_s,
    distance_m,
    transmission_delay_s,
)


@dataclass(slots=True)
class DecisionContext:
    task: Task
    vehicle: VehicleState
    vehicles: dict[str, VehicleState]
    service_quotes: Mapping[str, ServiceQuote]
    cloud_price: float
    cloud_queue_length: int
    cloud_platform_overhead_s: float
    server_position: tuple[float, float]
    price_history: tuple[float, ...]
    adjacency: Mapping[str, tuple[str, ...]]
    v2v_throughput_by_link: Mapping[str, Mapping[str, float]] | None = None
    service_spatial_index: ServiceSpatialIndex | None = None
    v2v_path_index: ReverseParetoV2VIndex | None = None
    cloud_capacity_ratio: float = 0.0
    cloud_target_offload_ratio: float = 0.0


def estimates(context: DecisionContext, config: SimulationConfig) -> dict[OffloadAction, OffloadEstimate]:
    task = context.task
    vehicle = context.vehicle
    transmitted_task, preprocessing_delay, preprocessing_energy = offload_preprocessing(
        task, vehicle, config
    )
    local_queue = vehicle.workload_cycles / vehicle.compute_hz
    local_compute = task.compute_cycles / vehicle.compute_hz
    local = OffloadEstimate(
        OffloadAction.LOCAL,
        delay_s=local_queue + local_compute,
        energy_j=local_compute * config.energy.local_compute_power_w,
        payment=0.0,
        feasible=vehicle.energy_level > 0.0,
        target_id=vehicle.vehicle_id,
        path=(vehicle.vehicle_id,),
    )
    v2v = (
        context.v2v_path_index.estimate(vehicle.vehicle_id, transmitted_task)
        if context.v2v_path_index is not None
        else best_v2v_path(
            vehicle.vehicle_id,
            context.vehicles,
            context.service_quotes,
            transmitted_task,
            config.network,
            context.adjacency,
            context.v2v_throughput_by_link,
            config.service_compute_hz,
            context.service_spatial_index,
            config.energy,
        )
    )
    if v2v.feasible:
        v2v = replace(
            v2v,
            delay_s=v2v.delay_s + preprocessing_delay,
            energy_j=v2v.energy_j + preprocessing_energy,
        )
    v2i = _v2i_estimate(
        context,
        config,
        transmitted_task,
        preprocessing_delay,
        preprocessing_energy,
    )
    return {item.action: item for item in (local, v2v, v2i)}


def v2i_estimate(
    context: DecisionContext,
    config: SimulationConfig,
) -> OffloadEstimate:
    """Estimate only V2I when repricing leaves Local and V2V physics unchanged."""
    transmitted_task, preprocessing_delay, preprocessing_energy = offload_preprocessing(
        context.task,
        context.vehicle,
        config,
    )
    return _v2i_estimate(
        context,
        config,
        transmitted_task,
        preprocessing_delay,
        preprocessing_energy,
    )


def _v2i_estimate(
    context: DecisionContext,
    config: SimulationConfig,
    transmitted_task: Task,
    preprocessing_delay: float,
    preprocessing_energy: float,
) -> OffloadEstimate:
    task = context.task
    vehicle = context.vehicle
    server_distance = distance_m(vehicle.position, context.server_position)
    v2i_tx = transmission_delay_s(
        transmitted_task.data_size_mb, server_distance, OffloadAction.V2I, config.network
    )
    v2i_compute = task.compute_cycles / config.cloud_compute_hz
    v2i_service = (
        v2i_compute
        + cloud_queue_delay_s(context.cloud_queue_length, config.network)
        + context.cloud_platform_overhead_s
    )
    v2i = OffloadEstimate(
        OffloadAction.V2I,
        delay_s=preprocessing_delay + v2i_tx + v2i_service,
        energy_j=(
            preprocessing_energy
            + v2i_tx * config.energy.v2i_transmit_power_w
            + v2i_compute * config.energy.cloud_compute_power_w
        ),
        payment=context.cloud_price * (task.compute_cycles / 1e9),
        target_id="cloud",
        path=(vehicle.vehicle_id, "cloud"),
    )
    return v2i


def offload_preprocessing(
    task: Task,
    vehicle: VehicleState,
    config: SimulationConfig,
) -> tuple[Task, float, float]:
    """Return the transmitted task plus local compression delay and energy."""
    ratio = config.offload_compression_ratio
    if ratio >= 1.0:
        return task, 0.0, 0.0
    compression_cycles = task.data_size_mb * config.compression_cycles_per_mb
    delay_s = compression_cycles / vehicle.compute_hz
    energy_j = delay_s * config.energy.local_compute_power_w
    return replace(task, data_size_mb=task.data_size_mb * ratio), delay_s, energy_j


def choose_action(
    strategy: str,
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
    policy_rng: random.Random,
    dqn: DQNAgent,
    explore: bool = True,
    precomputed_allowed_actions: list[int] | None = None,
    precomputed_state: np.ndarray | None = None,
    online_q_values: np.ndarray | None = None,
    target_q_values: np.ndarray | None = None,
    advance_exploration: bool = True,
    online_reliability: float = 1.0,
    game_adequacy: float = 1.0,
) -> tuple[OffloadAction, np.ndarray | None, bool]:
    feasible = [item for item in candidates.values() if item.feasible and math.isfinite(item.delay_s)]
    state = None
    if strategy in {"dqn", "hybrid_stackelberg"}:
        state = (
            precomputed_state
            if precomputed_state is not None
            else _state(context, candidates, config)
        )
    if not feasible:
        if strategy == "hybrid_stackelberg":
            dqn.advance_exploration(explore and advance_exploration)
        return OffloadAction.LOCAL, state, False
    on_time = [item for item in feasible if item.delay_s <= context.task.deadline_s]
    allowed_actions = (
        precomputed_allowed_actions
        if precomputed_allowed_actions is not None
        else policy_action_ids(strategy, context, candidates, config)
    )
    allowed_candidates = [candidates[OffloadAction(action)] for action in allowed_actions]
    used_dqn = False
    if strategy == "random":
        action = policy_rng.choice([item.action for item in feasible])
    elif strategy == "greedy":
        action = min(feasible, key=lambda item: (item.delay_s, int(item.action))).action
    elif strategy == "stackelberg":
        stackelberg_pool = (
            (on_time or feasible)
            if config.decision.stackelberg_deadline_action_masking
            else feasible
        )
        action = min(
            stackelberg_pool,
            key=lambda item: (_objective(item, context.task, config, context), int(item.action)),
        ).action
    elif strategy == "dqn":
        assert state is not None
        action = OffloadAction(
            dqn.choose_action_from_q(
                online_q_values,
                explore=explore,
                allowed_actions=allowed_actions,
                advance_exploration=advance_exploration,
                exploration_checked=False,
            )
            if online_q_values is not None
            else dqn.choose_action(
                state,
                explore=explore,
                allowed_actions=allowed_actions,
                advance_exploration=advance_exploration,
            )
        )
        used_dqn = True
    elif strategy == "hybrid_stackelberg":
        guidance = game_guidance(
            context,
            candidates,
            config,
            allowed_actions,
        )
        stackelberg_action = guidance.action
        game_confidence = guidance.confidence
        dominant = [
            candidate
            for candidate in allowed_candidates
            if _dominates_all(candidate, allowed_candidates)
        ]
        if len(dominant) == 1:
            action = dominant[0].action
        elif len(allowed_candidates) == 1:
            action = allowed_candidates[0].action
        elif config.decision.hybrid_fusion_mode == "adaptive_confidence":
            assert state is not None
            arbitration_q_values = (
                online_q_values
                if online_q_values is not None
                else dqn.q_values(state)
            )
            arbitration = hybrid_arbitration(
                context,
                guidance,
                allowed_actions,
                arbitration_q_values,
                config,
                dqn_reliability=(
                    1.0
                    if config.dqn.mode == "evaluate"
                    else min(
                        1.0,
                        dqn.transition_count
                        / max(config.dqn.warmup_transitions, 1),
                    )
                )
                * online_reliability,
                game_adequacy=game_adequacy,
            )
            if arbitration.use_game:
                action = stackelberg_action
            else:
                action = OffloadAction(
                    dqn.choose_action_from_q(
                        arbitration_q_values,
                        explore=explore,
                        allowed_actions=allowed_actions,
                        advance_exploration=advance_exploration,
                        exploration_checked=False,
                    )
                )
                used_dqn = True
        elif config.decision.hybrid_fusion_mode == "confidence_gated":
            if (
                len(dominant) == 1
                or game_confidence
                >= config.decision.hybrid_game_confidence_threshold
            ):
                action = stackelberg_action
            else:
                assert state is not None
                action = OffloadAction(
                    dqn.choose_action_from_q(
                        online_q_values,
                        explore=explore,
                        allowed_actions=allowed_actions,
                        advance_exploration=advance_exploration,
                        exploration_checked=False,
                    )
                    if online_q_values is not None
                    else dqn.choose_action(
                        state,
                        explore=explore,
                        allowed_actions=allowed_actions,
                        advance_exploration=advance_exploration,
                    )
                )
                used_dqn = True
        elif config.decision.hybrid_fusion_mode == "delegated":
            assert state is not None
            action = OffloadAction(
                dqn.choose_action_from_q(
                    online_q_values,
                    explore=explore,
                    allowed_actions=allowed_actions,
                    advance_exploration=advance_exploration,
                    exploration_checked=False,
                )
                if online_q_values is not None
                else dqn.choose_action(
                    state,
                    explore=explore,
                    allowed_actions=allowed_actions,
                    advance_exploration=advance_exploration,
                )
            )
            used_dqn = True
        else:
            assert state is not None
            immediate_rewards = {
                int(candidate.action): (
                    reward_for(candidate, context.task, config, context)[0]
                    if config.decision.hybrid_objective_guidance
                    else 0.0
                )
                for candidate in allowed_candidates
            }
            action = OffloadAction(
                dqn.choose_residual_action_from_q(
                    online_q_values,
                    target_q_values,
                    immediate_rewards,
                    _hybrid_residual_weight(context, config),
                    explore=explore,
                    allowed_actions=allowed_actions,
                    advance_exploration=advance_exploration,
                    exploration_checked=False,
                )
                if online_q_values is not None and target_q_values is not None
                else dqn.choose_residual_action(
                    state,
                    immediate_rewards,
                    _hybrid_residual_weight(context, config),
                    explore=explore,
                    allowed_actions=allowed_actions,
                    advance_exploration=advance_exploration,
                )
            )
            used_dqn = True
        if not used_dqn:
            # The Stackelberg gate and the DQN form one Hybrid decision.  The
            # exploration schedule therefore advances even when the gate can
            # resolve the current task without evaluating the network.
            dqn.advance_exploration(explore and advance_exploration)
    else:
        raise ValueError(f"unsupported strategy: {strategy}")
    if not candidates[action].feasible or not math.isfinite(candidates[action].delay_s):
        action = min(
            feasible,
            key=lambda item: (_objective(item, context.task, config, context), int(item.action)),
        ).action
    return action, state, used_dqn


def policy_action_ids(
    strategy: str,
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
) -> list[int]:
    feasible = [item for item in candidates.values() if item.feasible and math.isfinite(item.delay_s)]
    if not feasible:
        return [int(OffloadAction.LOCAL)]
    if strategy not in {"dqn", "hybrid_stackelberg"}:
        return [int(item.action) for item in feasible]

    safe = feasible
    if config.decision.deadline_action_masking:
        on_time = [item for item in feasible if item.delay_s <= context.task.deadline_s]
        if on_time:
            safe = on_time
    if strategy == "dqn" or not config.decision.hybrid_objective_guidance:
        return [int(item.action) for item in safe]

    safe = _capacity_aware_hybrid_candidates(context, safe, config)
    return [int(item.action) for item in safe]


def hybrid_decision_source(
    action: OffloadAction,
    used_dqn: bool,
    candidates: dict[OffloadAction, OffloadEstimate],
    allowed_actions: list[int] | tuple[int, ...],
) -> str:
    if used_dqn:
        return "dqn"
    allowed = [
        candidates[OffloadAction(action_id)]
        for action_id in allowed_actions
        if candidates[OffloadAction(action_id)].feasible
        and math.isfinite(candidates[OffloadAction(action_id)].delay_s)
    ]
    if len(allowed) == 1:
        return "single_feasible"
    dominant = [candidate for candidate in allowed if _dominates_all(candidate, allowed)]
    if len(dominant) == 1 and dominant[0].action == action:
        return "strict_dominance"
    return "game_gate"


def _capacity_aware_hybrid_candidates(
    context: DecisionContext,
    candidates: list[OffloadEstimate],
    config: SimulationConfig,
) -> list[OffloadEstimate]:
    """Reserve saturated cloud capacity for tasks without an on-time edge option."""
    decision = config.decision
    if not decision.hybrid_cloud_capacity_guard:
        return candidates
    queue_ratio = context.cloud_queue_length / max(config.network.queue_delay_threshold, 1)
    if queue_ratio < decision.hybrid_cloud_guard_ratio:
        return candidates
    on_time_non_cloud = [
        item
        for item in candidates
        if item.action != OffloadAction.V2I and item.delay_s <= context.task.deadline_s
    ]
    return on_time_non_cloud or candidates


def _hybrid_residual_weight(
    context: DecisionContext,
    config: SimulationConfig,
) -> float:
    """Reduce learned residual authority smoothly as the shared cloud saturates."""
    decision = config.decision
    weight = decision.hybrid_residual_weight
    if not decision.hybrid_residual_congestion_adaptation:
        return weight
    queue_ratio = context.cloud_queue_length / max(config.network.queue_delay_threshold, 1)
    start = decision.hybrid_residual_decay_start_ratio
    if queue_ratio <= start:
        return weight
    span = max(1.0 - start, 1e-9)
    progress = min(1.0, (queue_ratio - start) / span)
    scale = 1.0 - progress * (1.0 - decision.hybrid_residual_min_scale)
    return weight * scale


def reward_for(
    estimate: OffloadEstimate,
    task: Task,
    config: SimulationConfig,
    context: DecisionContext | None = None,
) -> tuple[float, bool]:
    success = estimate.delay_s <= task.deadline_s
    cost = _generalized_cost(estimate, task, config, context)
    if success:
        reward = config.reward.on_time_bonus - config.reward.cost_scale * cost
    else:
        reward = -config.reward.timeout_penalty - config.reward.cost_scale * cost
    return reward, success


def _state(
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
) -> np.ndarray:
    if hasattr(context.adjacency, "neighbor_count"):
        neighbors = context.adjacency.neighbor_count(context.vehicle.vehicle_id, limit=10)
    else:
        neighbors = len(context.adjacency.get(context.vehicle.vehicle_id, ()))
    v2v_target_id = candidates[OffloadAction.V2V].target_id
    v2v_target = context.vehicles.get(v2v_target_id or "")
    v2v_target_workload_ratio = (
        v2v_target.workload_cycles
        / max(v2v_target.compute_hz, 1.0)
        / max(context.task.deadline_s, 1e-12)
        if v2v_target is not None
        else 2.0
    )
    return build_state(
        context.task,
        context.vehicle,
        context.cloud_price,
        neighbors,
        context.cloud_queue_length / max(config.network.queue_delay_threshold, 1),
        context.cloud_capacity_ratio,
        v2v_target_workload_ratio,
        candidates,
        config.task_compute_max_cycles,
        config.task_data_max_mb,
        (
            max(config.task_deadlines_s)
            if config.task_deadline_distribution == "discrete"
            else config.task_deadline_max_s
        ),
        config.decision.energy_scale_j,
        config.decision.payment_scale,
        config.cloud_max_price,
    )


def decision_state(
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
) -> np.ndarray:
    """Build the public 20-dimensional policy state for batched inference."""
    return _state(context, candidates, config)


def stackelberg_best_action(
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
) -> OffloadAction:
    feasible = [item for item in candidates.values() if item.feasible and math.isfinite(item.delay_s)]
    if not feasible:
        return OffloadAction.LOCAL
    on_time = [item for item in feasible if item.delay_s <= context.task.deadline_s]
    pool = (
        (on_time or feasible)
        if config.decision.stackelberg_deadline_action_masking
        else feasible
    )
    return min(
        pool,
        key=lambda item: (_objective(item, context.task, config, context), int(item.action)),
    ).action


@dataclass(slots=True, frozen=True)
class GameGuidance:
    action: OffloadAction
    confidence: float


@dataclass(slots=True, frozen=True)
class HybridArbitration:
    """Dimensionless evidence used to select the game or learned response."""

    use_game: bool
    dqn_action: OffloadAction
    game_evidence: float
    dqn_evidence: float
    q_opposition: float
    cloud_pressure: float


def hybrid_arbitration(
    context: DecisionContext,
    guidance: GameGuidance,
    allowed_actions: list[int] | tuple[int, ...],
    q_values: np.ndarray,
    config: SimulationConfig,
    *,
    dqn_reliability: float = 1.0,
    game_adequacy: float = 1.0,
) -> HybridArbitration:
    """Balance immediate follower evidence against learned long-term advantage.

    The game margin and the DQN opposition are normalized independently.  When
    the two experts disagree about using the shared cloud, the evidence for the
    non-cloud action is strengthened in proportion to predicted overload.  This
    internalizes the queue externality without replacing each vehicle's
    autonomous DQN decision with a central assignment.
    """
    allowed = tuple(sorted({int(action) for action in allowed_actions}))
    if not allowed:
        allowed = (int(guidance.action),)
    values = np.asarray(q_values, dtype=np.float64)
    dqn_action_id = max(
        allowed,
        key=lambda action: (float(values[action]), -action),
    )
    dqn_action = OffloadAction(dqn_action_id)
    allowed_values = np.asarray([values[action] for action in allowed])
    q_range = float(np.max(allowed_values) - np.min(allowed_values))
    q_opposition = max(
        0.0,
        float(values[dqn_action_id] - values[int(guidance.action)])
        / max(q_range, 1e-12),
    )
    decision = config.decision
    game_evidence = guidance.confidence / max(
        decision.hybrid_game_confidence_threshold,
        1e-12,
    )
    # The follower margin grows without bound as congestion inflates cost
    # differences, while the Q opposition is range-normalized.  Damping the
    # game evidence by its demonstrated adequacy hands authority to the
    # learned expert exactly when following the game demonstrably fails.
    game_evidence *= (
        min(max(game_adequacy, 0.0), 1.0)
        ** decision.hybrid_adequacy_game_exponent
    )
    dqn_evidence = (
        min(max(dqn_reliability, 0.0), 1.0)
        * q_opposition
        / decision.hybrid_dqn_opposition_threshold
    )
    utilization_target = config.serverless.capacity_utilization_target
    cloud_pressure = max(
        0.0,
        context.cloud_capacity_ratio / max(utilization_target, 1e-12) - 1.0,
    )
    externality_multiplier = (
        1.0 + decision.hybrid_congestion_sensitivity * cloud_pressure
    )
    if guidance.action == OffloadAction.V2I and dqn_action != OffloadAction.V2I:
        dqn_evidence *= externality_multiplier
    elif guidance.action != OffloadAction.V2I and dqn_action == OffloadAction.V2I:
        game_evidence *= externality_multiplier
    if decision.hybrid_game_evidence_cap > 0.0:
        # Congestion inflates the follower margin without bound while the Q
        # opposition is range-normalized; the cap keeps both evidences
        # commensurable without tracking realized outcomes.
        game_evidence = min(game_evidence, decision.hybrid_game_evidence_cap)
    return HybridArbitration(
        use_game=(
            guidance.action == dqn_action
            or game_evidence >= dqn_evidence
        ),
        dqn_action=dqn_action,
        game_evidence=game_evidence,
        dqn_evidence=dqn_evidence,
        q_opposition=q_opposition,
        cloud_pressure=cloud_pressure,
    )


def game_guidance(
    context: DecisionContext,
    candidates: dict[OffloadAction, OffloadEstimate],
    config: SimulationConfig,
    allowed_actions: list[int] | tuple[int, ...] | None = None,
) -> GameGuidance:
    """Return the follower's finite-utility best response and its cost margin."""
    allowed = (
        {OffloadAction(action) for action in allowed_actions}
        if allowed_actions is not None
        else set(candidates)
    )
    pool = [
        item
        for item in candidates.values()
        if item.action in allowed and item.feasible and math.isfinite(item.delay_s)
    ]
    if not pool:
        return GameGuidance(OffloadAction.LOCAL, 1.0)
    ranked = sorted(
        (
            (_objective(item, context.task, config, context), int(item.action), item)
            for item in pool
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(ranked) == 1:
        return GameGuidance(ranked[0][2].action, 1.0)
    best_cost = ranked[0][0]
    second_cost = ranked[1][0]
    confidence = max(0.0, second_cost - best_cost) / max(abs(best_cost), 1.0)
    if _dominates_all(ranked[0][2], pool):
        confidence = max(confidence, 1.0)
    return GameGuidance(ranked[0][2].action, confidence)


def _objective(
    candidate: OffloadEstimate,
    task: Task,
    config: SimulationConfig,
    context: DecisionContext | None = None,
) -> float:
    decision = config.decision
    on_time_bonus = (
        decision.stackelberg_on_time_bonus * (1.0 + task.urgency)
        if candidate.delay_s <= task.deadline_s
        else 0.0
    )
    return _generalized_cost(candidate, task, config, context) - on_time_bonus


def _generalized_cost(
    candidate: OffloadEstimate,
    task: Task,
    config: SimulationConfig,
    context: DecisionContext | None = None,
) -> float:
    """Dimensionless immediate cost shared by Stackelberg and DQN rewards."""
    decision = config.decision
    return (
        decision.delay_weight * candidate.delay_s / task.deadline_s
        + decision.energy_weight * candidate.energy_j / decision.energy_scale_j
        + decision.payment_weight * candidate.payment / decision.payment_scale
    )


def _dominates_all(candidate: OffloadEstimate, options: list[OffloadEstimate]) -> bool:
    for other in options:
        if other is candidate:
            continue
        values = (candidate.delay_s, candidate.energy_j, candidate.payment)
        other_values = (other.delay_s, other.energy_j, other.payment)
        if not (all(a <= b for a, b in zip(values, other_values)) and any(a < b for a, b in zip(values, other_values))):
            return False
    return True
