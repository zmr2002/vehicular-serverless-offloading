from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Any


class OffloadAction(IntEnum):
    LOCAL = 0
    V2V = 1
    V2I = 2


@dataclass(slots=True, frozen=True)
class Task:
    task_id: str
    vehicle_id: str
    compute_cycles: float
    data_size_mb: float
    deadline_s: float
    urgency: float
    created_step: int


@dataclass(slots=True)
class VehicleState:
    vehicle_id: str
    position: tuple[float, float]
    speed_mps: float
    compute_hz: float
    energy_level: float = 1.0
    queue_length: int = 0
    workload_cycles: float = 0.0
    is_service: bool = False


@dataclass(slots=True, frozen=True)
class ServiceQuote:
    vehicle_id: str
    price: float
    compute_hz: float
    utility: float


@dataclass(slots=True, frozen=True)
class OffloadEstimate:
    action: OffloadAction
    delay_s: float
    energy_j: float
    payment: float
    feasible: bool = True
    target_id: str | None = None
    path: tuple[str, ...] = ()


@dataclass(slots=True)
class OffloadResult:
    task_id: str
    vehicle_id: str
    action: OffloadAction
    delay_s: float
    energy_j: float
    payment: float
    reward: float
    success: bool
    step: int
    target_id: str | None = None
    path: tuple[str, ...] = ()
    dispatch_queue_ms: float | None = None
    http_latency_ms: float | None = None
    client_latency_ms: float | None = None
    processing_ms: float | None = None
    platform_overhead_ms: float | None = None
    preprocessing_delay_ms: float | None = None
    radio_delay_ms: float | None = None
    physical_compute_ms: float | None = None
    physical_queue_ms: float | None = None
    scaled_processing_ms: float | None = None
    total_delay_ms: float | None = None
    http_attempts: int | None = None
    http_retry_count: int | None = None
    retry_backoff_ms: float | None = None
    cold_start: bool | None = None
    instance_id: str | None = None
    checksum: str | None = None
    server_distance_m: float | None = None
    oracle_action: OffloadAction | None = None
    oracle_delay_s: float | None = None
    decision_regret_s: float | None = None
    oracle_success: bool | None = None
    task_compute_cycles: float | None = None
    task_data_size_mb: float | None = None
    effective_offload_data_mb: float | None = None
    task_deadline_s: float | None = None
    local_estimate_s: float | None = None
    v2v_estimate_s: float | None = None
    v2i_estimate_s: float | None = None
    allowed_action_count: int | None = None
    used_dqn: bool | None = None
    stackelberg_action: OffloadAction | None = None
    hybrid_deviation: bool | None = None
    hybrid_deviation_beneficial: bool | None = None
    all_actions_late: bool | None = None
    dqn_deviation: bool | None = None
    rule_deviation: bool | None = None
    source_workload_s: float | None = None
    v2v_target_workload_s: float | None = None
    max_service_workload_s: float | None = None
    cloud_queue_length: int | None = None
    predicted_cloud_capacity_ratio: float | None = None
    cloud_target_offload_ratio: float | None = None
    q_local: float | None = None
    q_v2v: float | None = None
    q_v2i: float | None = None
    dqn_action: OffloadAction | None = None
    dqn_q_margin: float | None = None
    cloud_price: float | None = None
    game_action: OffloadAction | None = None
    game_confidence: float | None = None
    hybrid_game_evidence: float | None = None
    hybrid_dqn_evidence: float | None = None
    hybrid_q_opposition: float | None = None
    hybrid_cloud_pressure: float | None = None
    hybrid_decision_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "task_id": self.task_id,
            "vehicle_id": self.vehicle_id,
            "action": self.action.name.lower(),
            "delay_s": self.delay_s,
            "energy_j": self.energy_j,
            "payment": self.payment,
            "reward": self.reward,
            "success": int(self.success),
            "target_id": self.target_id or "",
            "path": ">".join(self.path),
            "dispatch_queue_ms": self.dispatch_queue_ms,
            "http_latency_ms": self.http_latency_ms,
            "client_latency_ms": self.client_latency_ms,
            "processing_ms": self.processing_ms,
            "platform_overhead_ms": self.platform_overhead_ms,
            "preprocessing_delay_ms": self.preprocessing_delay_ms,
            "radio_delay_ms": self.radio_delay_ms,
            "physical_compute_ms": self.physical_compute_ms,
            "physical_queue_ms": self.physical_queue_ms,
            "scaled_processing_ms": self.scaled_processing_ms,
            "total_delay_ms": self.total_delay_ms,
            "http_attempts": self.http_attempts,
            "http_retry_count": self.http_retry_count,
            "retry_backoff_ms": self.retry_backoff_ms,
            "cold_start": self.cold_start,
            "instance_id": self.instance_id or "",
            "checksum": self.checksum or "",
            "server_distance_m": self.server_distance_m,
            "oracle_action": self.oracle_action.name.lower() if self.oracle_action is not None else "",
            "oracle_delay_s": self.oracle_delay_s,
            "decision_regret_s": self.decision_regret_s,
            "oracle_success": int(self.oracle_success) if self.oracle_success is not None else "",
            "task_compute_cycles": self.task_compute_cycles,
            "task_data_size_mb": self.task_data_size_mb,
            "effective_offload_data_mb": self.effective_offload_data_mb,
            "task_deadline_s": self.task_deadline_s,
            "local_estimate_s": _finite_or_blank(self.local_estimate_s),
            "v2v_estimate_s": _finite_or_blank(self.v2v_estimate_s),
            "v2i_estimate_s": _finite_or_blank(self.v2i_estimate_s),
            "allowed_action_count": self.allowed_action_count,
            "used_dqn": int(self.used_dqn) if self.used_dqn is not None else "",
            "stackelberg_action": (
                self.stackelberg_action.name.lower() if self.stackelberg_action is not None else ""
            ),
            "hybrid_deviation": (
                int(self.hybrid_deviation) if self.hybrid_deviation is not None else ""
            ),
            "hybrid_deviation_beneficial": (
                int(self.hybrid_deviation_beneficial)
                if self.hybrid_deviation_beneficial is not None
                else ""
            ),
            "all_actions_late": (
                int(self.all_actions_late) if self.all_actions_late is not None else ""
            ),
            "dqn_deviation": (
                int(self.dqn_deviation) if self.dqn_deviation is not None else ""
            ),
            "rule_deviation": (
                int(self.rule_deviation) if self.rule_deviation is not None else ""
            ),
            "source_workload_s": self.source_workload_s,
            "v2v_target_workload_s": self.v2v_target_workload_s,
            "max_service_workload_s": self.max_service_workload_s,
            "cloud_queue_length": self.cloud_queue_length,
            "predicted_cloud_capacity_ratio": self.predicted_cloud_capacity_ratio,
            "cloud_target_offload_ratio": self.cloud_target_offload_ratio,
            "q_local": self.q_local,
            "q_v2v": self.q_v2v,
            "q_v2i": self.q_v2i,
            "dqn_action": (
                self.dqn_action.name.lower() if self.dqn_action is not None else ""
            ),
            "dqn_q_margin": self.dqn_q_margin,
            "cloud_price": self.cloud_price,
            "game_action": (
                self.game_action.name.lower()
                if self.game_action is not None
                else ""
            ),
            "game_confidence": self.game_confidence,
            "hybrid_game_evidence": self.hybrid_game_evidence,
            "hybrid_dqn_evidence": self.hybrid_dqn_evidence,
            "hybrid_q_opposition": self.hybrid_q_opposition,
            "hybrid_cloud_pressure": self.hybrid_cloud_pressure,
            "hybrid_decision_source": self.hybrid_decision_source,
        }


def _finite_or_blank(value: float | None) -> float | str:
    return value if value is not None and math.isfinite(value) else ""
