from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any, TypeVar
import tomllib


@dataclass(slots=True)
class DQNConfig:
    state_size: int = 20
    action_size: int = 3
    hidden_sizes: tuple[int, int] = (256, 128)
    replay_capacity: int = 10_000
    replay_sampling: str = "ring"
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    epsilon_decay_per_step: bool = False
    target_update_interval: int = 100
    warmup_transitions: int = 128
    training_interval: int = 4
    gradient_clip_norm: float = 10.0
    huber_delta: float = 1.0
    game_guidance_weight: float = 0.0
    game_guidance_margin: float = 0.1
    intraop_threads: int = 1
    interop_threads: int = 1
    device: str = "cpu"
    mode: str = "train"
    checkpoint_path: str | None = None


@dataclass(slots=True)
class NetworkConfig:
    # B[MHz] * spectral_efficiency[bit/s/Hz] * resource_efficiency = R[Mbit/s].
    # V2V has less reference spectrum and a larger unmodelled-interference loss
    # than the infrastructure-scheduled V2I link.
    v2v_channel_bandwidth_mhz: float = 120.0
    v2i_channel_bandwidth_mhz: float = 200.0
    v2v_resource_efficiency: float = 0.75
    v2i_resource_efficiency: float = 0.85
    neighbor_radius_m: float = 500.0
    max_hops: int = 3
    queue_delay_coefficient: float = 0.1
    queue_delay_threshold: int = 10
    queue_delay_extra_s: float = 0.2
    analytical_cold_start_s: float = 0.1
    channel_capacity_model: str = "shannon"
    v2v_reference_snr_db: float = 40.0
    v2i_reference_snr_db: float = 42.0
    v2v_reference_distance_m: float = 100.0
    v2i_reference_distance_m: float = 100.0
    v2v_path_loss_exponent: float = 3.5
    v2i_path_loss_exponent: float = 3.0
    minimum_channel_distance_m: float = 1.0
    v2v_max_spectral_efficiency_bps_hz: float = 10.0
    v2i_max_spectral_efficiency_bps_hz: float = 10.0


@dataclass(slots=True)
class RewardConfig:
    cost_scale: float = 10.0
    timeout_penalty: float = 200.0
    on_time_bonus: float = 30.0


@dataclass(slots=True)
class EnergyConfig:
    """Power coefficients used by the thesis energy equations."""

    local_compute_power_w: float = 100.0
    v2v_transmit_power_w: float = 50.0
    service_compute_power_w: float = 80.0
    v2i_transmit_power_w: float = 20.0
    cloud_compute_power_w: float = 80.0


@dataclass(slots=True)
class DecisionConfig:
    delay_weight: float = 1.0
    energy_weight: float = 0.2
    payment_weight: float = 0.075
    energy_scale_j: float = 250.0
    payment_scale: float = 0.5
    hybrid_residual_weight: float = 0.75
    hybrid_fusion_mode: str = "residual"
    hybrid_residual_congestion_adaptation: bool = False
    hybrid_residual_decay_start_ratio: float = 0.5
    hybrid_residual_min_scale: float = 0.2
    deadline_action_masking: bool = True
    stackelberg_deadline_action_masking: bool = False
    stackelberg_on_time_bonus: float = 0.0
    hybrid_objective_guidance: bool = True
    hybrid_cloud_capacity_guard: bool = False
    hybrid_cloud_guard_ratio: float = 1.0
    hybrid_game_confidence_threshold: float = 0.15
    hybrid_dqn_opposition_threshold: float = 0.15
    hybrid_congestion_sensitivity: float = 1.0
    hybrid_online_reliability: str = "off"
    hybrid_reliability_decay: float = 0.995
    hybrid_reliability_floor: float = 0.0
    hybrid_game_adequacy_arbitration: str = "off"
    hybrid_adequacy_defense_floor: float = 0.99
    hybrid_adequacy_defense_full: float = 0.999
    hybrid_adequacy_game_exponent: float = 8.0
    hybrid_game_evidence_cap: float = 0.0
    synchronous_v2v_queue_forecast: bool = False


@dataclass(slots=True)
class ServerlessConfig:
    endpoint: str = "http://127.0.0.1:8080"
    timeout_s: float = 30.0
    max_retries: int = 4
    retry_backoff_s: float = 0.05
    max_work_units: int = 25_000
    max_requests_per_run: int = 0
    idle_steps_to_zero: int = 50
    concurrency_target: int = 10
    max_instances: int = 10
    client_concurrency: int = 50
    capacity_utilization_target: float = 0.85


@dataclass(slots=True)
class SimulationConfig:
    steps: int = 2_000
    vehicle_count: int = 2_000
    seed: int = 42
    task_probability: float = 0.3
    strategy: str = "hybrid_stackelberg"
    decision_timing: str = "synchronous_batch"
    backend: str = "analytical"
    mobility: str = "synthetic"
    output_dir: str = "results/verified"
    record_decision_diagnostics: bool = False
    record_task_records: bool = True
    task_record_sample_rate: float = 1.0
    decision_trace_mode: str = "none"
    decision_trace_path: str | None = None
    minimum_free_disk_gb: float = 2.0
    scenario_config: str | None = None
    scenario_net: str | None = None
    route_output_dir: str = "scenarios/wakaba/generated"
    route_departure_end_s: float = 2_000.0
    sumo_binary: str = "sumo"
    vehicle_compute_hz: float = 2e9
    service_compute_hz: float = 2e9
    cloud_compute_hz: float = 50e9
    cloud_base_price: float = 0.1
    cloud_pricing_mode: str = "queue"
    cloud_target_offload_ratio: float = 0.3
    cloud_demand_sensitivity: float = 3.0
    cloud_price_smoothing: float = 0.1
    cloud_min_price: float = 0.05
    cloud_max_price: float = 1.0
    cloud_capacity_price_weight: float = 1.0
    cloud_price_candidate_count: int = 7
    cloud_price_response_temperature: float = 0.15
    cloud_price_response_iterations: int = 3
    cloud_price_response_min_iterations: int = 1
    cloud_price_response_relaxation: float = 0.5
    cloud_price_response_tolerance: float = 0.0
    cloud_price_response_policy: str = "softmax"
    cloud_price_state_consistency: bool = True
    cloud_price_batch_candidates: bool = True
    cloud_price_outer_iterations: int = 2
    cloud_price_outer_min_iterations: int = 2
    cloud_price_outer_tolerance: float = 0.0
    cloud_leader_timeout_weight: float = 0.5
    cloud_leader_late_tolerance: float = 0.02
    service_price_sensitivity: float = 0.5
    service_demand_price_weight: float = 0.15
    service_min_energy: float = 0.05
    service_max_queue: int = 5
    service_role_mode: str = "fixed_ratio"
    service_vehicle_ratio: float = 0.3
    vehicle_battery_capacity_j: float = 1.0e6
    service_vehicle_battery_capacity_j: float = 1.0e6
    task_compute_distribution: str = "uniform"
    task_compute_min_cycles: float = 1e9
    task_compute_max_cycles: float = 5e9
    task_compute_choices: tuple[float, ...] = (1e9, 5e9)
    task_data_min_mb: float = 1.0
    task_data_max_mb: float = 100.0
    offload_compression_ratio: float = 1.0
    compression_cycles_per_mb: float = 2.0e6
    task_deadline_distribution: str = "uniform"
    task_deadline_min_s: float = 2.0
    task_deadline_max_s: float = 3.0
    task_deadlines_s: tuple[float, ...] = (0.8, 1.5, 2.0)
    area_width_m: float = 5_000.0
    area_height_m: float = 5_000.0
    service_positions: tuple[tuple[float, float], ...] = (
        (1_083.33, 1_525.00),
        (1_650.00, 1_525.00),
        (2_216.67, 1_525.00),
        (1_083.33, 2_175.00),
        (1_650.00, 2_175.00),
        (2_216.67, 2_175.00),
    )
    dqn: DQNConfig = field(default_factory=DQNConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    serverless: ServerlessConfig = field(default_factory=ServerlessConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.steps <= 0 or self.vehicle_count <= 0:
            raise ValueError("steps and vehicle_count must be positive")
        if not 0.0 <= self.task_probability <= 1.0:
            raise ValueError("task_probability must be in [0, 1]")
        if self.minimum_free_disk_gb < 0.0:
            raise ValueError("minimum_free_disk_gb must be non-negative")
        if not 0.0 <= self.task_record_sample_rate <= 1.0:
            raise ValueError("task_record_sample_rate must be in [0, 1]")
        if self.decision_trace_mode not in {"none", "record", "replay"}:
            raise ValueError("decision_trace_mode must be none, record, or replay")
        if self.decision_trace_mode != "none" and not self.decision_trace_path:
            raise ValueError(
                "decision_trace_path is required for record or replay mode"
            )
        if self.strategy not in {"random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg"}:
            raise ValueError(f"unknown strategy: {self.strategy}")
        if self.decision_timing != "synchronous_batch":
            raise ValueError("decision_timing must be synchronous_batch")
        if self.backend not in {"analytical", "knative"}:
            raise ValueError(f"unknown backend: {self.backend}")
        if self.mobility not in {"synthetic", "sumo"}:
            raise ValueError(f"unknown mobility provider: {self.mobility}")
        if self.mobility == "sumo" and not (self.scenario_config or self.scenario_net):
            raise ValueError("scenario_config or scenario_net is required for SUMO mobility")
        if self.dqn.state_size != 20 or self.dqn.action_size != 3 or self.dqn.hidden_sizes != (256, 128):
            raise ValueError("the thesis DQN requires the 20->256->128->3 architecture")
        if self.dqn.replay_capacity < self.dqn.batch_size or self.dqn.batch_size <= 0:
            raise ValueError("DQN replay_capacity must be at least batch_size, and batch_size must be positive")
        if self.dqn.mode not in {"train", "evaluate"}:
            raise ValueError("DQN mode must be train or evaluate")
        if self.dqn.replay_sampling not in {"ring", "load_stratified"}:
            raise ValueError("DQN replay_sampling must be ring or load_stratified")
        if self.dqn.mode == "evaluate" and not self.dqn.checkpoint_path:
            raise ValueError("DQN evaluation requires checkpoint_path")
        if not 0.0 <= self.dqn.epsilon_end <= self.dqn.epsilon_start <= 1.0:
            raise ValueError("DQN epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < self.dqn.epsilon_decay <= 1.0 or not 0.0 <= self.dqn.gamma <= 1.0:
            raise ValueError("DQN epsilon_decay and gamma are outside their allowed ranges")
        if min(
            self.dqn.learning_rate,
            self.dqn.target_update_interval,
            self.dqn.training_interval,
            self.dqn.gradient_clip_norm,
            self.dqn.huber_delta,
            self.dqn.intraop_threads,
            self.dqn.interop_threads,
        ) <= 0:
            raise ValueError("DQN learning and stability settings must be positive")
        if self.dqn.game_guidance_weight < 0 or self.dqn.game_guidance_margin < 0:
            raise ValueError("DQN game guidance settings must be non-negative")
        if self.network.max_hops <= 0:
            raise ValueError("max_hops must be positive")
        if min(
            self.network.v2v_channel_bandwidth_mhz,
            self.network.v2i_channel_bandwidth_mhz,
        ) <= 0:
            raise ValueError("network channel bandwidths must be positive")
        if not 0.0 < self.network.v2v_resource_efficiency <= 1.0 or not 0.0 < (
            self.network.v2i_resource_efficiency
        ) <= 1.0:
            raise ValueError("network resource efficiencies must be in (0, 1]")
        if self.network.channel_capacity_model not in {"distance_only", "shannon"}:
            raise ValueError("channel_capacity_model must be distance_only or shannon")
        if min(
            self.network.v2v_path_loss_exponent,
            self.network.v2i_path_loss_exponent,
            self.network.v2v_reference_distance_m,
            self.network.v2i_reference_distance_m,
            self.network.minimum_channel_distance_m,
            self.network.v2v_max_spectral_efficiency_bps_hz,
            self.network.v2i_max_spectral_efficiency_bps_hz,
        ) <= 0:
            raise ValueError("channel capacity parameters must be positive")
        if self.network.neighbor_radius_m <= 0:
            raise ValueError("neighbor_radius_m must be positive")
        if min(self.vehicle_compute_hz, self.service_compute_hz, self.cloud_compute_hz) <= 0:
            raise ValueError("compute capacities must be positive")
        if min(
            self.energy.local_compute_power_w,
            self.energy.v2v_transmit_power_w,
            self.energy.service_compute_power_w,
            self.energy.v2i_transmit_power_w,
            self.energy.cloud_compute_power_w,
        ) <= 0:
            raise ValueError("energy power coefficients must be positive")
        if not 0.0 < self.cloud_target_offload_ratio <= 1.0:
            raise ValueError("cloud_target_offload_ratio must be in (0, 1]")
        if self.cloud_demand_sensitivity <= 0 or not 0.0 < self.cloud_price_smoothing <= 1.0:
            raise ValueError("cloud demand sensitivity and price smoothing are invalid")
        if not 0 <= self.cloud_min_price <= self.cloud_base_price <= self.cloud_max_price:
            raise ValueError("cloud prices must satisfy min <= base <= max")
        if self.cloud_pricing_mode not in {
            "queue",
            "leader_best_response",
            "follower_best_response",
        }:
            raise ValueError(
                "cloud_pricing_mode must be queue, leader_best_response, "
                "or follower_best_response"
            )
        if (
            self.cloud_capacity_price_weight < 0
            or self.cloud_leader_timeout_weight < 0
            or self.service_demand_price_weight < 0
        ):
            raise ValueError("pricing weights must be non-negative")
        if not 0.0 <= self.cloud_leader_late_tolerance <= 1.0:
            raise ValueError("cloud_leader_late_tolerance must be in [0, 1]")
        if self.cloud_price_candidate_count < 2:
            raise ValueError("cloud_price_candidate_count must be at least 2")
        if self.cloud_price_response_temperature <= 0:
            raise ValueError("cloud_price_response_temperature must be positive")
        if self.cloud_price_response_iterations <= 0:
            raise ValueError("cloud_price_response_iterations must be positive")
        if not 1 <= self.cloud_price_response_min_iterations <= self.cloud_price_response_iterations:
            raise ValueError(
                "cloud_price_response_min_iterations must be between 1 and "
                "cloud_price_response_iterations"
            )
        if not 0.0 < self.cloud_price_response_relaxation <= 1.0:
            raise ValueError("cloud_price_response_relaxation must be in (0, 1]")
        if self.cloud_price_response_tolerance < 0.0:
            raise ValueError("cloud_price_response_tolerance must be non-negative")
        if self.cloud_price_response_policy not in {"softmax", "argmax"}:
            raise ValueError(
                "cloud_price_response_policy must be softmax or argmax"
            )
        if self.cloud_price_outer_iterations <= 0:
            raise ValueError("cloud_price_outer_iterations must be positive")
        if not 1 <= self.cloud_price_outer_min_iterations <= self.cloud_price_outer_iterations:
            raise ValueError(
                "cloud_price_outer_min_iterations must be between 1 and "
                "cloud_price_outer_iterations"
            )
        if self.cloud_price_outer_tolerance < 0.0:
            raise ValueError("cloud_price_outer_tolerance must be non-negative")
        if min(self.vehicle_battery_capacity_j, self.service_vehicle_battery_capacity_j) <= 0:
            raise ValueError("battery capacities must be positive")
        if self.service_role_mode not in {"fixed_ratio", "dynamic_idle"}:
            raise ValueError("service_role_mode must be fixed_ratio or dynamic_idle")
        if self.service_role_mode == "dynamic_idle" and not math.isclose(
            self.service_compute_hz,
            self.vehicle_compute_hz,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "dynamic_idle uses the same physical vehicles in both roles, so "
                "service_compute_hz must equal vehicle_compute_hz"
            )
        if not 0.0 < self.service_vehicle_ratio < 1.0:
            raise ValueError("service_vehicle_ratio must be in (0, 1)")
        if self.task_compute_distribution not in {"uniform", "discrete"}:
            raise ValueError("task_compute_distribution must be uniform or discrete")
        if self.task_compute_min_cycles <= 0 or self.task_compute_max_cycles < self.task_compute_min_cycles:
            raise ValueError("task compute bounds are invalid")
        if self.task_compute_distribution == "discrete" and (
            not self.task_compute_choices or min(self.task_compute_choices) <= 0
        ):
            raise ValueError("task_compute_choices must contain positive values in discrete mode")
        if self.task_deadline_distribution not in {"uniform", "discrete"}:
            raise ValueError("task_deadline_distribution must be uniform or discrete")
        if (
            self.task_deadline_min_s <= 0
            or self.task_deadline_max_s < self.task_deadline_min_s
        ):
            raise ValueError("task deadline bounds are invalid")
        if self.task_deadline_distribution == "discrete" and (
            not self.task_deadlines_s or min(self.task_deadlines_s) <= 0
        ):
            raise ValueError("task_deadlines_s must contain positive values in discrete mode")
        if self.task_data_min_mb < 0 or self.task_data_max_mb < self.task_data_min_mb:
            raise ValueError("task data bounds are invalid")
        if not 0.0 < self.offload_compression_ratio <= 1.0:
            raise ValueError("offload_compression_ratio must be in (0, 1]")
        if self.compression_cycles_per_mb < 0:
            raise ValueError("compression_cycles_per_mb must be non-negative")
        if not self.service_positions:
            raise ValueError("at least one service position is required")
        if min(
            self.serverless.concurrency_target,
            self.serverless.max_instances,
            self.serverless.client_concurrency,
        ) <= 0:
            raise ValueError("serverless concurrency settings and max instances must be positive")
        if self.serverless.max_requests_per_run < 0:
            raise ValueError("serverless max_requests_per_run must be non-negative")
        if self.serverless.max_retries < 0 or self.serverless.max_retries > 10:
            raise ValueError("serverless max_retries must be between 0 and 10")
        if self.serverless.retry_backoff_s < 0:
            raise ValueError("serverless retry_backoff_s must be non-negative")
        if not 0.0 < self.serverless.capacity_utilization_target <= 1.0:
            raise ValueError("serverless capacity_utilization_target must be in (0, 1]")
        if min(self.reward.cost_scale, self.reward.timeout_penalty, self.reward.on_time_bonus) < 0:
            raise ValueError("reward coefficients must be non-negative")
        if min(self.decision.energy_scale_j, self.decision.payment_scale) <= 0:
            raise ValueError("decision normalization scales must be positive")
        if min(
            self.decision.delay_weight,
            self.decision.energy_weight,
            self.decision.payment_weight,
        ) < 0:
            raise ValueError("decision weights must be non-negative")
        if self.decision.hybrid_fusion_mode not in {
            "residual",
            "delegated",
            "confidence_gated",
            "adaptive_confidence",
        }:
            raise ValueError(
                "hybrid_fusion_mode must be residual, delegated, confidence_gated, "
                "or adaptive_confidence"
            )
        if self.decision.hybrid_residual_weight < 0:
            raise ValueError("hybrid_residual_weight must be non-negative")
        if not 0.0 <= self.decision.hybrid_residual_min_scale <= 1.0:
            raise ValueError("hybrid_residual_min_scale must be in [0, 1]")
        if min(
            self.decision.hybrid_residual_decay_start_ratio,
            self.decision.hybrid_cloud_guard_ratio,
            self.decision.stackelberg_on_time_bonus,
            self.decision.hybrid_game_confidence_threshold,
            self.decision.hybrid_congestion_sensitivity,
        ) < 0:
            raise ValueError("decision margins, bonuses, and congestion ratios must be non-negative")
        if self.decision.hybrid_dqn_opposition_threshold <= 0:
            raise ValueError("hybrid_dqn_opposition_threshold must be positive")
        if self.decision.hybrid_online_reliability not in {"off", "evaluate", "always"}:
            raise ValueError("hybrid_online_reliability must be off, evaluate, or always")
        if not 0.0 < self.decision.hybrid_reliability_decay <= 1.0:
            raise ValueError("hybrid_reliability_decay must be in (0, 1]")
        if not 0.0 <= self.decision.hybrid_reliability_floor <= 1.0:
            raise ValueError("hybrid_reliability_floor must be in [0, 1]")
        if self.decision.hybrid_game_adequacy_arbitration not in {"off", "evaluate", "always"}:
            raise ValueError(
                "hybrid_game_adequacy_arbitration must be off, evaluate, or always"
            )
        if not (
            0.0
            <= self.decision.hybrid_adequacy_defense_floor
            <= self.decision.hybrid_adequacy_defense_full
            <= 1.0
        ):
            raise ValueError(
                "adequacy defense thresholds must satisfy "
                "0 <= floor <= full <= 1"
            )
        if self.decision.hybrid_adequacy_game_exponent < 0.0:
            raise ValueError("hybrid_adequacy_game_exponent must be non-negative")
        if self.decision.hybrid_game_evidence_cap < 0.0:
            raise ValueError("hybrid_game_evidence_cap must be non-negative")
        if self.backend == "knative" and not self.serverless.endpoint.startswith(("http://", "https://")):
            raise ValueError("Knative endpoint must use HTTP or HTTPS")

    @classmethod
    def from_toml(cls, path: str | Path) -> "SimulationConfig":
        source = Path(path).resolve()
        raw = _load_toml_profile(source, ())
        root = raw.get("simulation", raw)
        nested = {
            "dqn": _construct(DQNConfig, raw.get("dqn", {})),
            "network": _construct(NetworkConfig, raw.get("network", {})),
            "energy": _construct(EnergyConfig, raw.get("energy", {})),
            "reward": _construct(RewardConfig, raw.get("reward", {})),
            "decision": _construct(DecisionConfig, raw.get("decision", {})),
            "serverless": _construct(ServerlessConfig, raw.get("serverless", {})),
        }
        config = _construct(cls, {**root, **nested})
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(kind: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(kind)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {kind.__name__} settings: {sorted(unknown)}")
    normalized = dict(values)
    for key in ("hidden_sizes", "task_compute_choices", "task_deadlines_s", "service_positions"):
        if key in normalized:
            normalized[key] = tuple(tuple(x) if isinstance(x, list) else x for x in normalized[key])
    return kind(**normalized)


def _load_toml_profile(
    source: Path,
    ancestry: tuple[Path, ...],
) -> dict[str, Any]:
    if source in ancestry:
        chain = " -> ".join(str(path) for path in (*ancestry, source))
        raise ValueError(f"cyclic TOML profile inheritance: {chain}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("extends must be a non-empty TOML path")
    parent_path = (source.parent / parent).resolve()
    inherited = _load_toml_profile(parent_path, (*ancestry, source))
    return _merge_profile(inherited, raw)


def _merge_profile(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _merge_profile(previous, value)
        else:
            merged[key] = value
    return merged
