from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.domain import OffloadAction
from vehicular_offloading.network import (
    cloud_queue_delay_s,
    effective_v2i_throughput_mbps,
    effective_v2v_throughput_mbps,
    spectral_efficiency_bps_hz,
)
from vehicular_offloading.serverless import (
    SERVERLESS_DELAY_MODEL,
    composed_service_delay_s,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check analytical load and radio invariants")
    parser.add_argument("--config", type=Path, default=Path("configs/paper-improved.toml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = SimulationConfig.from_toml(args.config)
    mean_cycles = _mean_cycles(config)
    mean_data_mb = (config.task_data_min_mb + config.task_data_max_mb) / 2.0
    mean_deadline_s = _mean_deadline(config)
    mean_service_s = mean_cycles / config.vehicle_compute_hz
    local_load = config.task_probability * mean_service_s
    critical_probability = min(1.0, 1.0 / mean_service_s)
    second_moment_s2 = _service_time_second_moment(config)
    queue_wait_s = (
        config.task_probability * second_moment_s2 / (2.0 * (1.0 - local_load))
        if local_load < 1.0
        else math.inf
    )

    reserve = config.serverless.capacity_utilization_target
    service_share = (
        reserve
        * (1.0 - config.task_probability)
        * config.service_compute_hz
        / (config.task_probability * mean_cycles)
    )
    communication_budget_s = mean_deadline_s - mean_service_s
    required_v2v_mbps = (
        mean_data_mb * 8.0 / communication_budget_s
        if communication_budget_s > 0.0
        else math.inf
    )
    edge_distance_m = config.network.neighbor_radius_m
    v2v_edge_mbps = effective_v2v_throughput_mbps(edge_distance_m, config.network)
    v2i_same_distance_mbps = effective_v2i_throughput_mbps(
        edge_distance_m, config.network
    )

    cloud_rows = []
    for active_vehicles in (1_000, 2_000, 4_000):
        tasks = config.task_probability * active_vehicles
        request_share = (
            reserve
            * config.serverless.max_instances
            * config.serverless.concurrency_target
            / tasks
        )
        compute_share = (
            reserve
            * config.serverless.max_instances
            * config.cloud_compute_hz
            / (tasks * mean_cycles)
        )
        cloud_rows.append(
            {
                "active_vehicles": active_vehicles,
                "request_limited_share": request_share,
                "compute_limited_share": compute_share,
                "capacity_target_share": min(
                    config.cloud_target_offload_ratio,
                    request_share,
                    compute_share,
                ),
            }
        )

    mean_cloud_compute_ms = mean_cycles / config.cloud_compute_hz * 1_000.0
    stressed_queue = config.network.queue_delay_threshold + 1
    stressed_queue_ms = (
        cloud_queue_delay_s(stressed_queue, config.network) * 1_000.0
    )
    composed_cloud_ms = (
        composed_service_delay_s(
            mean_cloud_compute_ms,
            stressed_queue_ms,
            dispatch_queue_ms=0.0,
            platform_overhead_ms=0.0,
        )
        * 1_000.0
    )
    checks = {
        "local_load_is_stressed_but_stable": 0.8 <= local_load < 1.0,
        "service_share_is_feasible": 0.0 < service_share < 1.0,
        "v2v_edge_rate_meets_mean_budget": v2v_edge_mbps >= required_v2v_mbps,
        "v2i_same_distance_is_stronger": v2i_same_distance_mbps > v2v_edge_mbps,
        "radio_units_are_distinct": (
            config.network.v2v_channel_bandwidth_mhz
            < config.network.v2i_channel_bandwidth_mhz
        ),
        "hybrid_fusion_mode_is_supported": config.decision.hybrid_fusion_mode
        in {"residual", "adaptive_confidence"},
        "decisions_use_synchronous_batch": (
            config.decision_timing == "synchronous_batch"
        ),
        "cloud_queue_grows_under_pressure": stressed_queue_ms > 0.0,
        "serverless_physical_components_compose_exactly": math.isclose(
            composed_cloud_ms,
            mean_cloud_compute_ms + stressed_queue_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }
    report = {
        "config": str(args.config),
        "task_model": {
            "mean_compute_cycles": mean_cycles,
            "mean_data_mb": mean_data_mb,
            "mean_deadline_s": mean_deadline_s,
            "mean_local_service_s": mean_service_s,
            "local_offered_load_ratio": local_load,
            "critical_task_probability": critical_probability,
            "mg1_queue_wait_approx_s": queue_wait_s,
        },
        "service_capacity": {
            "utilization_reserve": reserve,
            "v2v_task_share_at_reserve": service_share,
        },
        "radio": {
            "design_distance_m": edge_distance_m,
            "required_mean_v2v_throughput_mbps": required_v2v_mbps,
            "v2v_spectral_efficiency_bps_hz": spectral_efficiency_bps_hz(
                edge_distance_m, OffloadAction.V2V, config.network
            ),
            "v2v_effective_throughput_mbps": v2v_edge_mbps,
            "v2i_effective_throughput_mbps_at_same_distance": v2i_same_distance_mbps,
        },
        "decision": {
            "effective_payment_coefficient": (
                config.decision.payment_weight / config.decision.payment_scale
            ),
            "reward_cost_scale": config.reward.cost_scale,
            "hybrid_fusion_mode": config.decision.hybrid_fusion_mode,
            "hybrid_residual_weight": config.decision.hybrid_residual_weight,
            "decision_timing": config.decision_timing,
            "policy_sharing": "shared_parameters_private_vehicle_transitions",
        },
        "cloud_capacity": cloud_rows,
        "serverless": {
            "delay_model": SERVERLESS_DELAY_MODEL,
            "mean_physical_compute_ms": mean_cloud_compute_ms,
            "stressed_queue_length": stressed_queue,
            "stressed_physical_queue_ms": stressed_queue_ms,
            "scaled_function_processing_role": "diagnostic_only",
        },
        "checks": checks,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(checks.values()) else 1


def _mean_cycles(config: SimulationConfig) -> float:
    if config.task_compute_distribution == "discrete":
        return sum(config.task_compute_choices) / len(config.task_compute_choices)
    return (config.task_compute_min_cycles + config.task_compute_max_cycles) / 2.0


def _mean_deadline(config: SimulationConfig) -> float:
    if config.task_deadline_distribution == "discrete":
        return sum(config.task_deadlines_s) / len(config.task_deadlines_s)
    return (config.task_deadline_min_s + config.task_deadline_max_s) / 2.0


def _service_time_second_moment(config: SimulationConfig) -> float:
    if config.task_compute_distribution == "discrete":
        return sum(
            (cycles / config.vehicle_compute_hz) ** 2
            for cycles in config.task_compute_choices
        ) / len(config.task_compute_choices)
    lower = config.task_compute_min_cycles / config.vehicle_compute_hz
    upper = config.task_compute_max_cycles / config.vehicle_compute_hz
    return (lower * lower + lower * upper + upper * upper) / 3.0


if __name__ == "__main__":
    raise SystemExit(main())
