from __future__ import annotations

from array import array
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess

from .config import SimulationConfig
from .domain import OffloadAction, OffloadResult
from .serverless import SERVERLESS_DELAY_MODEL


TASK_COLUMNS = [
    "step", "task_id", "vehicle_id", "action", "delay_s", "energy_j", "payment", "reward",
    "success", "target_id", "path", "dispatch_queue_ms", "http_latency_ms",
    "client_latency_ms", "processing_ms", "platform_overhead_ms", "cold_start",
    "preprocessing_delay_ms", "radio_delay_ms", "physical_compute_ms",
    "physical_queue_ms", "scaled_processing_ms", "total_delay_ms",
    "http_attempts", "http_retry_count", "retry_backoff_ms", "instance_id", "checksum",
    "server_distance_m", "oracle_action", "oracle_delay_s", "decision_regret_s", "oracle_success",
    "task_compute_cycles", "task_data_size_mb", "effective_offload_data_mb", "task_deadline_s", "local_estimate_s",
    "v2v_estimate_s", "v2i_estimate_s", "allowed_action_count", "used_dqn",
    "stackelberg_action", "hybrid_deviation", "hybrid_deviation_beneficial",
    "all_actions_late", "dqn_deviation", "rule_deviation",
    "source_workload_s", "v2v_target_workload_s", "max_service_workload_s", "cloud_queue_length",
    "predicted_cloud_capacity_ratio", "cloud_target_offload_ratio",
    "q_local", "q_v2v", "q_v2i", "dqn_action", "dqn_q_margin",
    "cloud_price", "game_action", "game_confidence",
    "hybrid_game_evidence", "hybrid_dqn_evidence",
    "hybrid_q_opposition", "hybrid_cloud_pressure",
    "hybrid_decision_source",
]


@dataclass(slots=True)
class RunSummary:
    strategy: str
    backend: str
    mobility: str
    seed: int
    configured_steps: int
    completed_steps: int
    configured_vehicle_count: int
    realized_vehicle_count: int
    peak_active_vehicles: int
    total_tasks: int
    success_rate: float
    avg_energy_j: float
    avg_latency_s: float
    avg_success_latency_s: float
    total_cost: float
    avg_cost_per_task: float
    avg_reward: float
    local_offload_ratio: float
    v2v_offload_ratio: float
    v2i_offload_ratio: float
    local_success_rate: float
    v2v_success_rate: float
    v2i_success_rate: float
    oracle_success_rate: float
    avoidable_failure_rate: float
    avg_decision_regret_s: float
    avg_server_distance_m: float
    dqn_decision_ratio: float
    avg_allowed_action_count: float
    hybrid_deviation_ratio: float
    hybrid_beneficial_deviation_rate: float
    avg_hybrid_game_evidence: float
    avg_hybrid_dqn_evidence: float
    avg_hybrid_q_opposition: float
    avg_hybrid_cloud_pressure: float
    hybrid_strict_dominance_ratio: float
    hybrid_single_feasible_ratio: float
    hybrid_game_gate_ratio: float
    all_actions_late_rate: float
    all_late_cloud_admission_rate: float
    avg_all_late_cloud_cycles_per_step: float
    all_late_cloud_to_capacity_ratio: float
    dqn_deviation_ratio: float
    rule_deviation_ratio: float
    avg_cloud_queue_length: float
    max_cloud_queue_length: int
    avg_predicted_cloud_capacity_ratio: float
    max_predicted_cloud_capacity_ratio: float
    avg_cloud_target_offload_ratio: float
    avg_active_vehicle_count: float
    task_vehicle_step_ratio: float
    service_vehicle_step_ratio: float
    offered_vehicle_compute_load_ratio: float
    avg_source_workload_s: float
    p95_source_workload_s: float
    avg_v2v_target_workload_s: float
    p95_v2v_target_workload_s: float
    reachable_v2v_task_ratio: float
    v2v_latency_advantage_ratio: float
    queue_induced_local_timeout_ratio: float
    v2v_rescuable_task_ratio: float
    intrinsic_local_infeasible_task_ratio: float
    mandatory_remote_task_ratio: float
    avg_mandatory_remote_cycles_per_step: float
    mandatory_remote_to_cloud_capacity_ratio: float
    serverless_http_request_count: int
    serverless_http_attempt_count: int
    serverless_retried_request_count: int
    serverless_http_retry_count: int
    serverless_v2i_failure_count: int
    serverless_cold_start_count: int
    serverless_distinct_instance_count: int
    avg_serverless_client_latency_ms: float
    p95_serverless_client_latency_ms: float
    max_serverless_client_latency_ms: float
    max_serverless_cold_client_latency_ms: float
    p95_serverless_warm_client_latency_ms: float
    p95_serverless_dispatch_queue_ms: float
    p95_serverless_http_latency_ms: float
    p95_serverless_platform_overhead_ms: float
    avg_serverless_physical_compute_ms: float
    avg_serverless_scaled_processing_ms: float
    serverless_delay_decomposition_max_error_ms: float
    dqn_transitions: int
    replay_size: int
    dqn_updates: int
    final_epsilon: float


class RunRecorder:
    def __init__(self, output_root: str | Path, config: SimulationConfig):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(output_root) / f"{stamp}-{config.strategy}-{config.vehicle_count}-seed{config.seed}"
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = self.run_dir.with_name(f"{self.run_dir.name}-{suffix}")
            suffix += 1
        self.run_dir.mkdir(parents=True)
        self._handle = (self.run_dir / "tasks.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=TASK_COLUMNS)
        self._writer.writeheader()
        self._pricing_handle = (self.run_dir / "pricing.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._record_task_records = config.record_task_records
        self._task_record_sample_rate = config.task_record_sample_rate
        self._task_record_rng = random.Random(config.seed ^ 0x5EED5EED)
        self._task_records_seen = 0
        self._task_records_written = 0
        self._minimum_free_disk_bytes = int(config.minimum_free_disk_gb * (1024 ** 3))
        self._records_since_disk_check = 0
        self._config = config
        self._pricing_windows: dict[int, dict[str, float | int]] = {}
        self._check_free_disk()
        self.write_json("config.json", config.to_dict())
        self.write_json(
            "environment.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": _git_commit(),
                "serverless_delay_model": SERVERLESS_DELAY_MODEL,
            },
        )

    def record(self, result: OffloadResult) -> None:
        self._task_records_seen += 1
        should_write = (
            self._record_task_records
            and (
                self._task_record_sample_rate >= 1.0
                or (
                    self._task_record_sample_rate > 0.0
                    and self._task_record_rng.random() < self._task_record_sample_rate
                )
            )
        )
        if should_write:
            self._writer.writerow(result.to_row())
            self._task_records_written += 1
            self._records_since_disk_check += 1
            if self._records_since_disk_check >= 10_000:
                self._handle.flush()
                self._check_free_disk()
                self._records_since_disk_check = 0

    def record_pricing(self, value: dict) -> None:
        self._pricing_handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        window_index = int(value.get("step", 0)) // 250
        window = self._pricing_windows.setdefault(
            window_index,
            {
                "window_start_step": window_index * 250,
                "window_end_step": window_index * 250 + 249,
                "steps": 0,
                "tasks": 0,
                "price_sum": 0.0,
                "price_min": math.inf,
                "price_max": -math.inf,
                "prediction_weight": 0,
                "prediction_error_sum": 0.0,
                "prediction_abs_error_sum": 0.0,
                "response_residual_sum": 0.0,
                "outer_residual_sum": 0.0,
            },
        )
        tasks = int(value.get("task_count") or 0)
        price = float(value.get("selected_price") or 0.0)
        window["steps"] += 1
        window["tasks"] += tasks
        window["price_sum"] += price
        window["price_min"] = min(float(window["price_min"]), price)
        window["price_max"] = max(float(window["price_max"]), price)
        prediction_error = value.get("prediction_error")
        if prediction_error is not None:
            weight = max(tasks, 1)
            error = float(prediction_error)
            window["prediction_weight"] += weight
            window["prediction_error_sum"] += weight * error
            window["prediction_abs_error_sum"] += weight * abs(error)
        response_residual = value.get("response_cycle_residual")
        if response_residual is not None:
            window["response_residual_sum"] += float(response_residual)
        outer_residual = value.get("outer_cycle_residual")
        if outer_residual is not None:
            window["outer_residual_sum"] += float(outer_residual)

    def finish(self, summary: RunSummary) -> Path:
        self._handle.close()
        self._pricing_handle.close()
        self.write_json(
            "task-recording.json",
            {
                "enabled": self._record_task_records,
                "sample_rate": self._task_record_sample_rate,
                "records_seen": self._task_records_seen,
                "records_written": self._task_records_written,
            },
        )
        self.write_json("summary.json", asdict(summary))
        self.write_json("pricing-diagnostics.json", self._pricing_diagnostics())
        return self.run_dir

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._pricing_handle.closed:
            self._pricing_handle.close()

    def write_json(self, name: str, value) -> None:
        (self.run_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def _check_free_disk(self) -> None:
        if self._minimum_free_disk_bytes <= 0:
            return
        free_bytes = shutil.disk_usage(self.run_dir).free
        if free_bytes < self._minimum_free_disk_bytes:
            free_gb = free_bytes / (1024 ** 3)
            required_gb = self._minimum_free_disk_bytes / (1024 ** 3)
            raise RuntimeError(
                f"result recording stopped with {free_gb:.2f} GiB free; "
                f"minimum_free_disk_gb is {required_gb:.2f}"
            )

    def _pricing_diagnostics(self) -> dict:
        windows = []
        for index in sorted(self._pricing_windows):
            raw = self._pricing_windows[index]
            steps = int(raw["steps"]) or 1
            weight = int(raw["prediction_weight"])
            windows.append(
                {
                    "window_index": index,
                    "window_start_step": raw["window_start_step"],
                    "window_end_step": raw["window_end_step"],
                    "steps": raw["steps"],
                    "tasks": raw["tasks"],
                    "avg_price": float(raw["price_sum"]) / steps,
                    "min_price": 0.0 if math.isinf(float(raw["price_min"])) else raw["price_min"],
                    "max_price": 0.0 if math.isinf(float(raw["price_max"])) else raw["price_max"],
                    "weighted_prediction_bias": (
                        float(raw["prediction_error_sum"]) / weight if weight else 0.0
                    ),
                    "weighted_prediction_mae": (
                        float(raw["prediction_abs_error_sum"]) / weight if weight else 0.0
                    ),
                    "avg_response_cycle_residual": float(raw["response_residual_sum"]) / steps,
                    "avg_outer_cycle_residual": float(raw["outer_residual_sum"]) / steps,
                }
            )
        return {"window_size_steps": 250, "windows": windows}


class MetricsAccumulator:
    def __init__(self, vehicle_compute_hz: float = 2.0e9) -> None:
        self.vehicle_compute_hz = vehicle_compute_hz
        self.total = 0
        self.successes = 0
        self.energy = 0.0
        self.latency = 0.0
        self.success_latency = 0.0
        self.cost = 0.0
        self.reward = 0.0
        self.actions = {action: 0 for action in OffloadAction}
        self.action_successes = {action: 0 for action in OffloadAction}
        self.oracle_successes = 0
        self.avoidable_failures = 0
        self.decision_regret = 0.0
        self.server_distance = 0.0
        self.dqn_decisions = 0
        self.allowed_actions = 0
        self.hybrid_deviations = 0
        self.hybrid_beneficial_deviations = 0
        self.hybrid_arbitrations = 0
        self.hybrid_game_evidence = 0.0
        self.hybrid_dqn_evidence = 0.0
        self.hybrid_q_opposition = 0.0
        self.hybrid_cloud_pressure = 0.0
        self.hybrid_decision_sources = {
            "strict_dominance": 0,
            "single_feasible": 0,
            "game_gate": 0,
        }
        self.all_actions_late = 0
        self.all_late_cloud_admissions = 0
        self.all_late_cloud_cycles = 0.0
        self.dqn_deviations = 0
        self.rule_deviations = 0
        self.cloud_queue_total = 0
        self.cloud_queue_max = 0
        self.cloud_capacity_total = 0.0
        self.cloud_capacity_max = 0.0
        self.cloud_target_total = 0.0
        self.active_vehicle_steps = 0
        self.task_vehicle_steps = 0
        self.service_vehicle_steps = 0
        self.arrived_cycles = 0.0
        self.source_workloads_s = array("d")
        self.v2v_target_workloads_s = array("d")
        self.reachable_v2v_tasks = 0
        self.v2v_latency_advantages = 0
        self.queue_induced_local_timeouts = 0
        self.v2v_rescuable_tasks = 0
        self.mandatory_remote_tasks = 0
        self.mandatory_remote_cycles = 0.0
        self.serverless_http_requests = 0
        self.serverless_http_attempts = 0
        self.serverless_retried_requests = 0
        self.serverless_http_retries = 0
        self.serverless_v2i_failures = 0
        self.serverless_cold_starts = 0
        self.serverless_instances: set[str] = set()
        self.serverless_client_latency_ms = array("d")
        self.serverless_cold_client_latency_ms = array("d")
        self.serverless_warm_client_latency_ms = array("d")
        self.serverless_dispatch_queue_ms = array("d")
        self.serverless_http_latency_ms = array("d")
        self.serverless_platform_overhead_ms = array("d")
        self.serverless_physical_compute_ms = array("d")
        self.serverless_scaled_processing_ms = array("d")
        self.serverless_delay_decomposition_error_ms = array("d")
        self.diagnostic_window_size = 250
        self.diagnostic_windows: dict[int, dict] = {}
        self.game_to_dqn = _action_matrix()
        self.game_to_final = _action_matrix()
        self.dqn_to_final = _action_matrix()

    def observe_step(
        self,
        active_vehicle_count: int,
        task_vehicle_count: int,
        service_vehicle_count: int,
        arrived_cycles: float,
        step: int | None = None,
    ) -> None:
        self.active_vehicle_steps += active_vehicle_count
        self.task_vehicle_steps += task_vehicle_count
        self.service_vehicle_steps += service_vehicle_count
        self.arrived_cycles += arrived_cycles
        if step is not None:
            window = self._diagnostic_window(step)
            window["observed_steps"] += 1
            window["active_vehicle_steps"] += active_vehicle_count
            window["task_vehicle_steps"] += task_vehicle_count
            window["service_vehicle_steps"] += service_vehicle_count
            window["arrived_cycles"] += arrived_cycles

    def add(self, result: OffloadResult) -> None:
        self.total += 1
        self.successes += int(result.success)
        self.energy += result.energy_j
        self.latency += result.delay_s
        if result.success:
            self.success_latency += result.delay_s
        self.cost += result.payment
        self.reward += result.reward
        self.actions[result.action] += 1
        self.action_successes[result.action] += int(result.success)
        self.oracle_successes += int(bool(result.oracle_success))
        self.avoidable_failures += int(bool(result.oracle_success) and not result.success)
        self.decision_regret += result.decision_regret_s or 0.0
        self.server_distance += result.server_distance_m or 0.0
        self.dqn_decisions += int(bool(result.used_dqn))
        self.allowed_actions += result.allowed_action_count or 0
        self.hybrid_deviations += int(bool(result.hybrid_deviation))
        self.hybrid_beneficial_deviations += int(bool(result.hybrid_deviation_beneficial))
        if result.hybrid_game_evidence is not None:
            self.hybrid_arbitrations += 1
            self.hybrid_game_evidence += result.hybrid_game_evidence
            self.hybrid_dqn_evidence += result.hybrid_dqn_evidence or 0.0
            self.hybrid_q_opposition += result.hybrid_q_opposition or 0.0
            self.hybrid_cloud_pressure += result.hybrid_cloud_pressure or 0.0
        if result.hybrid_decision_source in self.hybrid_decision_sources:
            self.hybrid_decision_sources[result.hybrid_decision_source] += 1
        self.all_actions_late += int(bool(result.all_actions_late))
        if result.all_actions_late and result.action == OffloadAction.V2I:
            self.all_late_cloud_admissions += 1
            self.all_late_cloud_cycles += result.task_compute_cycles or 0.0
        self.dqn_deviations += int(bool(result.dqn_deviation))
        self.rule_deviations += int(bool(result.rule_deviation))
        queue_length = result.cloud_queue_length or 0
        self.cloud_queue_total += queue_length
        self.cloud_queue_max = max(self.cloud_queue_max, queue_length)
        capacity_ratio = result.predicted_cloud_capacity_ratio or 0.0
        self.cloud_capacity_total += capacity_ratio
        self.cloud_capacity_max = max(self.cloud_capacity_max, capacity_ratio)
        self.cloud_target_total += result.cloud_target_offload_ratio or 0.0
        source_workload_s = result.source_workload_s or 0.0
        self.source_workloads_s.append(max(0.0, source_workload_s))
        if result.v2v_target_workload_s is not None:
            self.v2v_target_workloads_s.append(max(0.0, result.v2v_target_workload_s))
        compute_cycles = result.task_compute_cycles or 0.0
        deadline_s = result.task_deadline_s or 0.0
        local_delay_s = result.local_estimate_s if result.local_estimate_s is not None else math.inf
        v2v_delay_s = result.v2v_estimate_s if result.v2v_estimate_s is not None else math.inf
        intrinsic_local_infeasible = compute_cycles / self.vehicle_compute_hz > deadline_s
        if math.isfinite(v2v_delay_s):
            self.reachable_v2v_tasks += 1
            self.v2v_latency_advantages += int(v2v_delay_s < local_delay_s)
            self.v2v_rescuable_tasks += int(
                local_delay_s > deadline_s and v2v_delay_s <= deadline_s
            )
        self.queue_induced_local_timeouts += int(
            not intrinsic_local_infeasible and local_delay_s > deadline_s
        )
        if intrinsic_local_infeasible:
            self.mandatory_remote_tasks += 1
            self.mandatory_remote_cycles += compute_cycles
        self._add_serverless_result(result)
        self._add_diagnostic_result(result)

    def _add_serverless_result(self, result: OffloadResult) -> None:
        attempts = result.http_attempts or 0
        if attempts <= 0:
            return
        self.serverless_http_requests += 1
        self.serverless_http_attempts += attempts
        retries = result.http_retry_count or 0
        self.serverless_retried_requests += int(retries > 0)
        self.serverless_http_retries += retries
        self.serverless_v2i_failures += int(not result.success)
        self.serverless_cold_starts += int(bool(result.cold_start))
        if result.instance_id:
            self.serverless_instances.add(result.instance_id)
        _append_optional(self.serverless_client_latency_ms, result.client_latency_ms)
        _append_optional(self.serverless_dispatch_queue_ms, result.dispatch_queue_ms)
        _append_optional(self.serverless_http_latency_ms, result.http_latency_ms)
        _append_optional(
            self.serverless_platform_overhead_ms,
            result.platform_overhead_ms,
        )
        _append_optional(
            self.serverless_physical_compute_ms,
            result.physical_compute_ms,
        )
        _append_optional(
            self.serverless_scaled_processing_ms,
            result.scaled_processing_ms,
        )
        if result.client_latency_ms is not None:
            target = (
                self.serverless_cold_client_latency_ms
                if result.cold_start
                else self.serverless_warm_client_latency_ms
            )
            target.append(float(result.client_latency_ms))
        components = (
            result.preprocessing_delay_ms,
            result.radio_delay_ms,
            result.physical_compute_ms,
            result.physical_queue_ms,
            result.dispatch_queue_ms,
            result.platform_overhead_ms,
        )
        if result.total_delay_ms is not None and all(
            value is not None for value in components
        ):
            self.serverless_delay_decomposition_error_ms.append(
                abs(
                    float(result.total_delay_ms)
                    - sum(float(value) for value in components if value is not None)
                )
            )

    def diagnostics(self, completed_steps: int) -> dict:
        return {
            "schema_version": 1,
            "window_size_steps": self.diagnostic_window_size,
            "completed_steps": completed_steps,
            "action_labels": [action.name.lower() for action in OffloadAction],
            "global_action_matrices": {
                "game_to_dqn": self.game_to_dqn,
                "game_to_final": self.game_to_final,
                "dqn_to_final": self.dqn_to_final,
            },
            "windows": [
                self._finalize_diagnostic_window(index, self.diagnostic_windows[index])
                for index in sorted(self.diagnostic_windows)
            ],
        }

    def _diagnostic_window(self, step: int) -> dict:
        index = max(0, int(step)) // self.diagnostic_window_size
        if index not in self.diagnostic_windows:
            self.diagnostic_windows[index] = {
                "task_count": 0,
                "successes": 0,
                "oracle_successes": 0,
                "avoidable_failures": 0,
                "reward_sum": 0.0,
                "latency_sum": 0.0,
                "energy_sum": 0.0,
                "cost_sum": 0.0,
                "actions": {action.name.lower(): 0 for action in OffloadAction},
                "action_successes": {action.name.lower(): 0 for action in OffloadAction},
                "cloud_queue_sum": 0.0,
                "cloud_queue_max": 0,
                "cloud_capacity_sum": 0.0,
                "cloud_price_sum": 0.0,
                "cloud_price_count": 0,
                "all_actions_late": 0,
                "dqn_decisions": 0,
                "hybrid_deviations": 0,
                "beneficial_deviations": 0,
                "q_margin_sum": 0.0,
                "q_margin_count": 0,
                "decision_sources": {},
                "game_to_dqn": _action_matrix(),
                "game_to_final": _action_matrix(),
                "dqn_to_final": _action_matrix(),
                "observed_steps": 0,
                "active_vehicle_steps": 0,
                "task_vehicle_steps": 0,
                "service_vehicle_steps": 0,
                "arrived_cycles": 0.0,
            }
        return self.diagnostic_windows[index]

    def _add_diagnostic_result(self, result: OffloadResult) -> None:
        window = self._diagnostic_window(result.step)
        action_name = result.action.name.lower()
        window["task_count"] += 1
        window["successes"] += int(result.success)
        window["oracle_successes"] += int(bool(result.oracle_success))
        window["avoidable_failures"] += int(bool(result.oracle_success) and not result.success)
        window["reward_sum"] += result.reward
        window["latency_sum"] += result.delay_s
        window["energy_sum"] += result.energy_j
        window["cost_sum"] += result.payment
        window["actions"][action_name] += 1
        window["action_successes"][action_name] += int(result.success)
        queue = result.cloud_queue_length or 0
        window["cloud_queue_sum"] += queue
        window["cloud_queue_max"] = max(window["cloud_queue_max"], queue)
        window["cloud_capacity_sum"] += result.predicted_cloud_capacity_ratio or 0.0
        if result.cloud_price is not None:
            window["cloud_price_sum"] += result.cloud_price
            window["cloud_price_count"] += 1
        window["all_actions_late"] += int(bool(result.all_actions_late))
        window["dqn_decisions"] += int(bool(result.used_dqn))
        window["hybrid_deviations"] += int(bool(result.hybrid_deviation))
        window["beneficial_deviations"] += int(bool(result.hybrid_deviation_beneficial))
        if result.dqn_q_margin is not None:
            window["q_margin_sum"] += result.dqn_q_margin
            window["q_margin_count"] += 1
        source = result.hybrid_decision_source or "none"
        window["decision_sources"][source] = window["decision_sources"].get(source, 0) + 1
        game_action = (
            result.game_action
            if result.game_action is not None
            else result.stackelberg_action
        )
        dqn_action = result.dqn_action
        if game_action is not None and dqn_action is not None:
            _increment_matrix(self.game_to_dqn, game_action, dqn_action)
            _increment_matrix(window["game_to_dqn"], game_action, dqn_action)
        if game_action is not None:
            _increment_matrix(self.game_to_final, game_action, result.action)
            _increment_matrix(window["game_to_final"], game_action, result.action)
        if dqn_action is not None:
            _increment_matrix(self.dqn_to_final, dqn_action, result.action)
            _increment_matrix(window["dqn_to_final"], dqn_action, result.action)

    def _finalize_diagnostic_window(self, index: int, raw: dict) -> dict:
        tasks = raw["task_count"] or 1
        steps = raw["observed_steps"] or 1
        q_count = raw["q_margin_count"] or 1
        price_count = raw["cloud_price_count"] or 1
        return {
            "window_index": index,
            "start_step": index * self.diagnostic_window_size,
            "end_step": index * self.diagnostic_window_size + max(raw["observed_steps"] - 1, 0),
            "observed_steps": raw["observed_steps"],
            "task_count": raw["task_count"],
            "success_rate": raw["successes"] / tasks,
            "oracle_success_rate": raw["oracle_successes"] / tasks,
            "avoidable_failure_rate": raw["avoidable_failures"] / tasks,
            "avg_reward": raw["reward_sum"] / tasks,
            "avg_latency_s": raw["latency_sum"] / tasks,
            "avg_energy_j": raw["energy_sum"] / tasks,
            "avg_cost_per_task": raw["cost_sum"] / tasks,
            "action_ratios": {key: value / tasks for key, value in raw["actions"].items()},
            "action_success_rates": {
                key: raw["action_successes"][key] / value if value else 0.0
                for key, value in raw["actions"].items()
            },
            "avg_cloud_queue_length": raw["cloud_queue_sum"] / tasks,
            "max_cloud_queue_length": raw["cloud_queue_max"],
            "avg_predicted_cloud_capacity_ratio": raw["cloud_capacity_sum"] / tasks,
            "avg_cloud_price": raw["cloud_price_sum"] / price_count,
            "all_actions_late_rate": raw["all_actions_late"] / tasks,
            "dqn_decision_ratio": raw["dqn_decisions"] / tasks,
            "hybrid_deviation_ratio": raw["hybrid_deviations"] / tasks,
            "beneficial_deviation_rate": (
                raw["beneficial_deviations"] / raw["hybrid_deviations"]
                if raw["hybrid_deviations"]
                else 0.0
            ),
            "avg_dqn_q_margin": raw["q_margin_sum"] / q_count,
            "decision_sources": raw["decision_sources"],
            "game_to_dqn": raw["game_to_dqn"],
            "game_to_final": raw["game_to_final"],
            "dqn_to_final": raw["dqn_to_final"],
            "avg_active_vehicles": raw["active_vehicle_steps"] / steps,
            "task_vehicle_step_ratio": raw["task_vehicle_steps"] / max(raw["active_vehicle_steps"], 1),
            "service_vehicle_step_ratio": raw["service_vehicle_steps"] / max(raw["active_vehicle_steps"], 1),
            "arrived_cycles": raw["arrived_cycles"],
        }

    def summary(
        self,
        config: SimulationConfig,
        completed_steps: int,
        realized_vehicle_count: int,
        peak_active_vehicles: int,
        replay_size: int,
        dqn_transitions: int,
        dqn_updates: int,
        final_epsilon: float,
    ) -> RunSummary:
        divisor = self.total or 1
        active_divisor = self.active_vehicle_steps or 1
        target_workload_count = len(self.v2v_target_workloads_s)
        return RunSummary(
            strategy=config.strategy,
            backend=config.backend,
            mobility=config.mobility,
            seed=config.seed,
            configured_steps=config.steps,
            completed_steps=completed_steps,
            configured_vehicle_count=config.vehicle_count,
            realized_vehicle_count=realized_vehicle_count,
            peak_active_vehicles=peak_active_vehicles,
            total_tasks=self.total,
            success_rate=self.successes / divisor,
            avg_energy_j=self.energy / divisor,
            avg_latency_s=self.latency / divisor,
            avg_success_latency_s=self.success_latency / self.successes if self.successes else 0.0,
            total_cost=self.cost,
            avg_cost_per_task=self.cost / divisor,
            avg_reward=self.reward / divisor,
            local_offload_ratio=self.actions[OffloadAction.LOCAL] / divisor,
            v2v_offload_ratio=self.actions[OffloadAction.V2V] / divisor,
            v2i_offload_ratio=self.actions[OffloadAction.V2I] / divisor,
            local_success_rate=(
                self.action_successes[OffloadAction.LOCAL] / self.actions[OffloadAction.LOCAL]
                if self.actions[OffloadAction.LOCAL]
                else 0.0
            ),
            v2v_success_rate=(
                self.action_successes[OffloadAction.V2V] / self.actions[OffloadAction.V2V]
                if self.actions[OffloadAction.V2V]
                else 0.0
            ),
            v2i_success_rate=(
                self.action_successes[OffloadAction.V2I] / self.actions[OffloadAction.V2I]
                if self.actions[OffloadAction.V2I]
                else 0.0
            ),
            oracle_success_rate=self.oracle_successes / divisor,
            avoidable_failure_rate=self.avoidable_failures / divisor,
            avg_decision_regret_s=self.decision_regret / divisor,
            avg_server_distance_m=self.server_distance / divisor,
            dqn_decision_ratio=self.dqn_decisions / divisor,
            avg_allowed_action_count=self.allowed_actions / divisor,
            hybrid_deviation_ratio=self.hybrid_deviations / divisor,
            hybrid_beneficial_deviation_rate=(
                self.hybrid_beneficial_deviations / self.hybrid_deviations
                if self.hybrid_deviations
                else 0.0
            ),
            avg_hybrid_game_evidence=(
                self.hybrid_game_evidence / self.hybrid_arbitrations
                if self.hybrid_arbitrations
                else 0.0
            ),
            avg_hybrid_dqn_evidence=(
                self.hybrid_dqn_evidence / self.hybrid_arbitrations
                if self.hybrid_arbitrations
                else 0.0
            ),
            avg_hybrid_q_opposition=(
                self.hybrid_q_opposition / self.hybrid_arbitrations
                if self.hybrid_arbitrations
                else 0.0
            ),
            avg_hybrid_cloud_pressure=(
                self.hybrid_cloud_pressure / self.hybrid_arbitrations
                if self.hybrid_arbitrations
                else 0.0
            ),
            hybrid_strict_dominance_ratio=(
                self.hybrid_decision_sources["strict_dominance"] / divisor
            ),
            hybrid_single_feasible_ratio=(
                self.hybrid_decision_sources["single_feasible"] / divisor
            ),
            hybrid_game_gate_ratio=(
                self.hybrid_decision_sources["game_gate"] / divisor
            ),
            all_actions_late_rate=self.all_actions_late / divisor,
            all_late_cloud_admission_rate=(
                self.all_late_cloud_admissions / self.all_actions_late
                if self.all_actions_late
                else 0.0
            ),
            avg_all_late_cloud_cycles_per_step=(
                self.all_late_cloud_cycles / max(completed_steps, 1)
            ),
            all_late_cloud_to_capacity_ratio=(
                self.all_late_cloud_cycles
                / max(completed_steps, 1)
                / (config.cloud_compute_hz * config.serverless.max_instances)
            ),
            dqn_deviation_ratio=self.dqn_deviations / divisor,
            rule_deviation_ratio=self.rule_deviations / divisor,
            avg_cloud_queue_length=self.cloud_queue_total / divisor,
            max_cloud_queue_length=self.cloud_queue_max,
            avg_predicted_cloud_capacity_ratio=self.cloud_capacity_total / divisor,
            max_predicted_cloud_capacity_ratio=self.cloud_capacity_max,
            avg_cloud_target_offload_ratio=self.cloud_target_total / divisor,
            avg_active_vehicle_count=self.active_vehicle_steps / max(completed_steps, 1),
            task_vehicle_step_ratio=self.task_vehicle_steps / active_divisor,
            service_vehicle_step_ratio=self.service_vehicle_steps / active_divisor,
            offered_vehicle_compute_load_ratio=(
                self.arrived_cycles / (active_divisor * self.vehicle_compute_hz)
            ),
            avg_source_workload_s=sum(self.source_workloads_s) / divisor,
            p95_source_workload_s=_percentile(self.source_workloads_s, 0.95),
            avg_v2v_target_workload_s=(
                sum(self.v2v_target_workloads_s) / target_workload_count
                if target_workload_count
                else 0.0
            ),
            p95_v2v_target_workload_s=_percentile(self.v2v_target_workloads_s, 0.95),
            reachable_v2v_task_ratio=self.reachable_v2v_tasks / divisor,
            v2v_latency_advantage_ratio=self.v2v_latency_advantages / divisor,
            queue_induced_local_timeout_ratio=self.queue_induced_local_timeouts / divisor,
            v2v_rescuable_task_ratio=self.v2v_rescuable_tasks / divisor,
            intrinsic_local_infeasible_task_ratio=self.mandatory_remote_tasks / divisor,
            mandatory_remote_task_ratio=self.mandatory_remote_tasks / divisor,
            avg_mandatory_remote_cycles_per_step=(
                self.mandatory_remote_cycles / max(completed_steps, 1)
            ),
            mandatory_remote_to_cloud_capacity_ratio=(
                self.mandatory_remote_cycles
                / max(completed_steps, 1)
                / (config.cloud_compute_hz * config.serverless.max_instances)
            ),
            serverless_http_request_count=self.serverless_http_requests,
            serverless_http_attempt_count=self.serverless_http_attempts,
            serverless_retried_request_count=self.serverless_retried_requests,
            serverless_http_retry_count=self.serverless_http_retries,
            serverless_v2i_failure_count=self.serverless_v2i_failures,
            serverless_cold_start_count=self.serverless_cold_starts,
            serverless_distinct_instance_count=len(self.serverless_instances),
            avg_serverless_client_latency_ms=_average(
                self.serverless_client_latency_ms
            ),
            p95_serverless_client_latency_ms=_percentile(
                self.serverless_client_latency_ms,
                0.95,
            ),
            max_serverless_client_latency_ms=max(
                self.serverless_client_latency_ms,
                default=0.0,
            ),
            max_serverless_cold_client_latency_ms=max(
                self.serverless_cold_client_latency_ms,
                default=0.0,
            ),
            p95_serverless_warm_client_latency_ms=_percentile(
                self.serverless_warm_client_latency_ms,
                0.95,
            ),
            p95_serverless_dispatch_queue_ms=_percentile(
                self.serverless_dispatch_queue_ms,
                0.95,
            ),
            p95_serverless_http_latency_ms=_percentile(
                self.serverless_http_latency_ms,
                0.95,
            ),
            p95_serverless_platform_overhead_ms=_percentile(
                self.serverless_platform_overhead_ms,
                0.95,
            ),
            avg_serverless_physical_compute_ms=_average(
                self.serverless_physical_compute_ms
            ),
            avg_serverless_scaled_processing_ms=_average(
                self.serverless_scaled_processing_ms
            ),
            serverless_delay_decomposition_max_error_ms=max(
                self.serverless_delay_decomposition_error_ms,
                default=0.0,
            ),
            dqn_transitions=dqn_transitions,
            replay_size=replay_size,
            dqn_updates=dqn_updates,
            final_epsilon=final_epsilon,
        )


def _percentile(values: array, quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _average(values: array) -> float:
    return sum(values) / len(values) if values else 0.0


def _append_optional(values: array, value: float | None) -> None:
    if value is not None and math.isfinite(float(value)):
        values.append(float(value))


def _action_matrix() -> dict[str, dict[str, int]]:
    labels = [action.name.lower() for action in OffloadAction]
    return {source: {target: 0 for target in labels} for source in labels}


def _increment_matrix(
    matrix: dict[str, dict[str, int]],
    source: OffloadAction,
    target: OffloadAction,
) -> None:
    matrix[source.name.lower()][target.name.lower()] += 1


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
