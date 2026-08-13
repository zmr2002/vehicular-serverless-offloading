from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.simulation import SimulationRunner


PROFILE_FIELDS = (
    "payload_max_mb",
    "compression_ratio",
    "compression_cycles_per_mb",
    "delay_weight",
    "energy_weight",
    "payment_weight",
    "hybrid_fusion_mode",
    "capacity_price_weight",
    "training_interval",
    "intraop_threads",
    "max_instances",
    "cloud_compute_hz",
    "v2v_channel_bandwidth_mhz",
    "v2v_reference_snr_db",
    "v2v_path_loss_exponent",
    "v2i_path_loss_exponent",
    "v2v_max_spectral_efficiency_bps_hz",
    "deadline_action_masking",
    "hybrid_objective_guidance",
    "hybrid_cloud_capacity_guard",
)


def apply_profile(config: SimulationConfig, profile: dict) -> None:
    if "payload_max_mb" in profile:
        config.task_data_max_mb = float(profile["payload_max_mb"])
    if "compression_ratio" in profile:
        config.offload_compression_ratio = float(profile["compression_ratio"])
    if "compression_cycles_per_mb" in profile:
        config.compression_cycles_per_mb = float(profile["compression_cycles_per_mb"])
    if "delay_weight" in profile:
        config.decision.delay_weight = float(profile["delay_weight"])
    if "energy_weight" in profile:
        config.decision.energy_weight = float(profile["energy_weight"])
    if "payment_weight" in profile:
        config.decision.payment_weight = float(profile["payment_weight"])
    if "hybrid_fusion_mode" in profile:
        config.decision.hybrid_fusion_mode = str(profile["hybrid_fusion_mode"])
    if "capacity_price_weight" in profile:
        config.cloud_capacity_price_weight = float(profile["capacity_price_weight"])
    if "training_interval" in profile:
        config.dqn.training_interval = int(profile["training_interval"])
    if "intraop_threads" in profile:
        config.dqn.intraop_threads = int(profile["intraop_threads"])
    if "max_instances" in profile:
        config.serverless.max_instances = int(profile["max_instances"])
    if "cloud_compute_hz" in profile:
        config.cloud_compute_hz = float(profile["cloud_compute_hz"])
    if "v2v_channel_bandwidth_mhz" in profile:
        config.network.v2v_channel_bandwidth_mhz = float(
            profile["v2v_channel_bandwidth_mhz"]
        )
    if "v2v_reference_snr_db" in profile:
        config.network.v2v_reference_snr_db = float(profile["v2v_reference_snr_db"])
    if "v2v_path_loss_exponent" in profile:
        config.network.v2v_path_loss_exponent = float(profile["v2v_path_loss_exponent"])
    if "v2i_path_loss_exponent" in profile:
        config.network.v2i_path_loss_exponent = float(profile["v2i_path_loss_exponent"])
    if "v2v_max_spectral_efficiency_bps_hz" in profile:
        config.network.v2v_max_spectral_efficiency_bps_hz = float(
            profile["v2v_max_spectral_efficiency_bps_hz"]
        )
    if "deadline_action_masking" in profile:
        config.decision.deadline_action_masking = bool(profile["deadline_action_masking"])
    if "hybrid_objective_guidance" in profile:
        config.decision.hybrid_objective_guidance = bool(profile["hybrid_objective_guidance"])
    if "hybrid_cloud_capacity_guard" in profile:
        config.decision.hybrid_cloud_capacity_guard = bool(profile["hybrid_cloud_capacity_guard"])


def effective_profile(config: SimulationConfig) -> dict:
    return {
        "payload_max_mb": config.task_data_max_mb,
        "compression_ratio": config.offload_compression_ratio,
        "compression_cycles_per_mb": config.compression_cycles_per_mb,
        "delay_weight": config.decision.delay_weight,
        "energy_weight": config.decision.energy_weight,
        "payment_weight": config.decision.payment_weight,
        "hybrid_fusion_mode": config.decision.hybrid_fusion_mode,
        "capacity_price_weight": config.cloud_capacity_price_weight,
        "training_interval": config.dqn.training_interval,
        "intraop_threads": config.dqn.intraop_threads,
        "max_instances": config.serverless.max_instances,
        "cloud_compute_hz": config.cloud_compute_hz,
        "v2v_channel_bandwidth_mhz": config.network.v2v_channel_bandwidth_mhz,
        "v2v_reference_snr_db": config.network.v2v_reference_snr_db,
        "v2v_path_loss_exponent": config.network.v2v_path_loss_exponent,
        "v2i_path_loss_exponent": config.network.v2i_path_loss_exponent,
        "v2v_max_spectral_efficiency_bps_hz": (
            config.network.v2v_max_spectral_efficiency_bps_hz
        ),
        "deadline_action_masking": config.decision.deadline_action_masking,
        "hybrid_objective_guidance": config.decision.hybrid_objective_guidance,
        "hybrid_cloud_capacity_guard": config.decision.hybrid_cloud_capacity_guard,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_completed(path: Path) -> dict[tuple[str, int, str], dict]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    completed = {}
    for row in rows:
        run_dir = Path(row["run_dir"])
        if (run_dir / "summary.json").exists() and (run_dir / "timing.json").exists():
            completed[(row["profile"], int(row["configured_vehicle_count"]), row["strategy"])] = row
    return completed


def execute(config: SimulationConfig, profile_name: str) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    wall_s = perf_counter() - wall_started
    cpu_s = process_time() - cpu_started
    timing_path = Path(run_dir) / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["wall_clock_s"] = wall_s
    timing["process_cpu_s"] = cpu_s
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return {
        "profile": profile_name,
        **effective_profile(config),
        **asdict(summary),
        "wall_clock_s": wall_s,
        "process_cpu_s": cpu_s,
        **{f"phase_{name}_s": value for name, value in timing["phase_seconds"].items()},
        "run_dir": run_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable thesis optimization profiles")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        raw = tomllib.load(handle)
    sweep = raw["sweep"]
    profiles = raw["profile"]
    base_path = (args.config.parent / sweep["base_config"]).resolve()
    base = SimulationConfig.from_toml(base_path)
    output = (args.config.parent.parent / sweep["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "optimization-progress.csv"
    completed = load_completed(progress_path)
    rows = list(completed.values())
    reference_strategies = sweep.get("reference_strategies", [])
    total = len(sweep["vehicle_counts"]) * (len(reference_strategies) + len(profiles))

    if args.dry_run:
        checked = 0
        for vehicles in sweep["vehicle_counts"]:
            cases = [
                (strategy, profiles[0]) for strategy in reference_strategies
            ] + [
                (profile.get("strategy", "hybrid_stackelberg"), profile)
                for profile in profiles
            ]
            for strategy, profile in cases:
                config = copy.deepcopy(base)
                apply_profile(config, profile)
                config.strategy = strategy
                config.steps = int(sweep["steps"])
                config.vehicle_count = int(vehicles)
                config.seed = int(sweep["seed"])
                config.validate()
                checked += 1
        print(f"DRY RUN OK: {checked}/{total} configurations validated")
        return 0

    base_profile = copy.deepcopy(profiles[0])
    for vehicles in sweep["vehicle_counts"]:
        cases = [
            (f"reference_{strategy}", strategy, base_profile)
            for strategy in reference_strategies
        ] + [
            (profile["name"], profile.get("strategy", "hybrid_stackelberg"), profile)
            for profile in profiles
        ]
        for profile_name, strategy, profile in cases:
            key = (profile_name, int(vehicles), strategy)
            if key in completed:
                print(f"SKIP {len(rows)}/{total} {profile_name} vehicles={vehicles}", flush=True)
                continue
            config = copy.deepcopy(base)
            apply_profile(config, profile)
            config.strategy = strategy
            config.steps = int(sweep["steps"])
            config.vehicle_count = int(vehicles)
            config.seed = int(sweep["seed"])
            config.output_dir = str(output / "runs" / profile_name)
            config.validate()
            print(f"START {len(rows) + 1}/{total} {profile_name} vehicles={vehicles}", flush=True)
            row = execute(config, profile_name)
            rows.append(row)
            write_rows(progress_path, rows)
            print(
                f"DONE {len(rows)}/{total} success={100 * float(row['success_rate']):.2f}% "
                f"wall={float(row['wall_clock_s']):.1f}s",
                flush=True,
            )
    write_rows(output / "optimization-results.csv", rows)
    (output / "optimization-results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output / "optimization-results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
