from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import TraceCachingMobilityProvider, create_mobility
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen follower-state and fixed-point corrections"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    repo = config_path.parent.parent
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    sweep = raw["sweep"]
    profiles = raw["profile"]
    parallelism = int(args.parallelism or sweep.get("parallelism", 6))
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    output = (repo / sweep["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        int(vehicles): _discover_checkpoint(
            repo / sweep["checkpoint_root"],
            int(vehicles),
        )
        for vehicles in sweep["vehicle_counts"]
    }
    cases = []
    for profile in profiles:
        base_path = (config_path.parent / profile["config"]).resolve()
        base = SimulationConfig.from_toml(base_path)
        for vehicles in sweep["vehicle_counts"]:
            for seed in sweep["seeds"]:
                config = copy.deepcopy(base)
                config.steps = int(sweep["steps"])
                config.vehicle_count = int(vehicles)
                config.seed = int(seed)
                config.strategy = "hybrid_stackelberg"
                config.backend = "analytical"
                config.output_dir = str(
                    output / "runs" / profile["name"] / f"v{vehicles}-seed{seed}"
                )
                config.dqn.mode = "evaluate"
                config.dqn.checkpoint_path = str(checkpoints[int(vehicles)])
                sample_rate = float(sweep.get("task_record_sample_rate", 0.001))
                config.record_task_records = sample_rate > 0.0
                config.task_record_sample_rate = sample_rate
                config.record_decision_diagnostics = True
                config.minimum_free_disk_gb = float(sweep["minimum_free_disk_gb"])
                config.validate()
                cases.append((profile["name"], config))
    profile_paths = [
        (config_path.parent / profile["config"]).resolve()
        for profile in profiles
    ]
    signature = _signature(config_path, checkpoints, profile_paths)
    state_path = output / "state.json"
    state = _load_state(state_path, signature)
    pending = []
    for index, (profile, config) in enumerate(cases, start=1):
        key = _case_key(profile, config)
        saved = state["runs"].get(key)
        if saved and _completed(Path(saved["run_dir"])):
            print(f"SKIP {index}/{len(cases)} {key}", flush=True)
        else:
            pending.append((index, profile, config))
    if args.dry_run:
        print(
            f"DRY RUN OK: {len(cases)} cases, {len(pending)} pending, "
            f"parallelism={parallelism}"
        )
        return 0
    _prepare_shared_inputs(pending)
    workers = min(parallelism, len(pending))
    if workers:
        scheduled = sorted(
            pending,
            key=lambda item: item[2].vehicle_count,
            reverse=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, profile, config in scheduled:
                print(
                    f"START {index}/{len(cases)} {profile} "
                    f"vehicles={config.vehicle_count} seed={config.seed}",
                    flush=True,
                )
                futures[executor.submit(_execute, profile, config)] = (
                    index,
                    profile,
                    config,
                )
            failures = []
            for future in as_completed(futures):
                index, profile, config = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    failures.append((index, profile, config, error))
                    print(f"FAILED {index}/{len(cases)} {error}", flush=True)
                    continue
                state["runs"][_case_key(profile, config)] = row
                _write_json_atomic(state_path, state)
                print(
                    f"DONE {index}/{len(cases)} success="
                    f"{100 * row['success_rate']:.2f}% bias="
                    f"{100 * row['pricing_weighted_bias']:+.2f}pp "
                    f"wall={row['wall_clock_s']:.1f}s",
                    flush=True,
                )
            if failures:
                raise RuntimeError(f"{len(failures)} ablation run(s) failed") from failures[0][3]
    rows = list(state["runs"].values())
    if len(rows) != len(cases):
        raise RuntimeError(f"expected {len(cases)} completed rows, found {len(rows)}")
    rows.sort(key=lambda row: (row["profile"], row["configured_vehicle_count"], row["seed"]))
    _write_csv(output / "run-results.csv", rows)
    aggregates = _aggregate(rows)
    _write_csv(output / "aggregate-results.csv", aggregates)
    selection = _select_profile(aggregates, profiles)
    _write_json_atomic(output / "selected-profile.json", selection)
    (output / "screening-summary.md").write_text(
        _markdown(aggregates, selection),
        encoding="utf-8",
    )
    print(f"COMPLETE {output}")
    print(f"SELECTED {selection['name']} ({selection['reason']})")
    return 0


def _execute(profile: str, config: SimulationConfig) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    pricing = _pricing_stats(Path(run_dir) / "pricing.jsonl")
    return {
        "profile": profile,
        **asdict(summary),
        **pricing,
        "wall_clock_s": perf_counter() - wall_started,
        "process_cpu_s": process_time() - cpu_started,
        "run_dir": run_dir,
    }


def _pricing_stats(path: Path) -> dict:
    weighted_error = 0.0
    weighted_abs_error = 0.0
    weight = 0.0
    iterations = []
    request_residuals = []
    cycle_residuals = []
    outer_iterations = []
    outer_request_residuals = []
    outer_cycle_residuals = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            error = row.get("prediction_error")
            tasks = float(row.get("task_count") or 0.0)
            if error is not None and tasks > 0:
                weighted_error += tasks * float(error)
                weighted_abs_error += tasks * abs(float(error))
                weight += tasks
            for key, target in (
                ("response_iterations", iterations),
                ("response_request_residual", request_residuals),
                ("response_cycle_residual", cycle_residuals),
                ("outer_iterations", outer_iterations),
                ("outer_request_residual", outer_request_residuals),
                ("outer_cycle_residual", outer_cycle_residuals),
            ):
                value = row.get(key)
                if value is not None and math.isfinite(float(value)):
                    target.append(float(value))
    return {
        "pricing_weighted_bias": weighted_error / max(weight, 1.0),
        "pricing_weighted_mae": weighted_abs_error / max(weight, 1.0),
        "response_iterations_mean": statistics.fmean(iterations) if iterations else 0.0,
        "response_request_residual_mean": (
            statistics.fmean(request_residuals) if request_residuals else 0.0
        ),
        "response_cycle_residual_mean": (
            statistics.fmean(cycle_residuals) if cycle_residuals else 0.0
        ),
        "outer_iterations_mean": (
            statistics.fmean(outer_iterations) if outer_iterations else 0.0
        ),
        "outer_request_residual_mean": (
            statistics.fmean(outer_request_residuals)
            if outer_request_residuals
            else 0.0
        ),
        "outer_cycle_residual_mean": (
            statistics.fmean(outer_cycle_residuals)
            if outer_cycle_residuals
            else 0.0
        ),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "success_rate",
        "avg_cloud_queue_length",
        "all_late_cloud_admission_rate",
        "all_late_cloud_to_capacity_ratio",
        "pricing_weighted_bias",
        "pricing_weighted_mae",
        "response_iterations_mean",
        "response_request_residual_mean",
        "response_cycle_residual_mean",
        "outer_iterations_mean",
        "outer_request_residual_mean",
        "outer_cycle_residual_mean",
        "wall_clock_s",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["profile"], int(row["configured_vehicle_count"])),
            [],
        ).append(row)
    result = []
    for (profile, vehicles), group in sorted(grouped.items()):
        item = {"profile": profile, "vehicle_count": vehicles, "replicates": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def _select_profile(aggregates: list[dict], profiles: list[dict]) -> dict:
    by_key = {(row["profile"], int(row["vehicle_count"])): row for row in aggregates}
    baseline = "current"
    candidates = []
    for profile in (item["name"] for item in profiles if item["name"] != "state_argmax_control"):
        rows = [by_key[(profile, vehicles)] for vehicles in (2000, 4000)]
        base_rows = [by_key[(baseline, vehicles)] for vehicles in (2000, 4000)]
        success_2000_gain = rows[0]["success_rate_mean"] - base_rows[0]["success_rate_mean"]
        success_4000_gain = rows[1]["success_rate_mean"] - base_rows[1]["success_rate_mean"]
        mae = statistics.fmean(row["pricing_weighted_mae_mean"] for row in rows)
        base_mae = statistics.fmean(row["pricing_weighted_mae_mean"] for row in base_rows)
        passes = (
            success_4000_gain >= -0.01
            and (mae <= 0.02 or mae <= 0.5 * base_mae)
            and success_2000_gain >= 0.0
        )
        score = statistics.fmean(row["success_rate_mean"] for row in rows) - mae
        candidates.append((passes, score, profile, success_2000_gain, success_4000_gain, mae))
    passing = [item for item in candidates if item[0]]
    selected = max(passing, default=None, key=lambda item: (item[1], item[2]))
    if selected is None:
        return {
            "name": baseline,
            "config": "paper-thesis-hybrid.toml",
            "promoted": False,
            "reason": "no correction passed the bias and non-regression gates",
        }
    config_by_name = {item["name"]: item["config"] for item in profiles}
    return {
        "name": selected[2],
        "config": config_by_name[selected[2]],
        "promoted": selected[2] != baseline,
        "success_2000_gain": selected[3],
        "success_4000_gain": selected[4],
        "pricing_mae": selected[5],
        "reason": "highest gated mean success minus pricing MAE",
    }


def _markdown(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Follower-state correction screening",
        "",
        "| Profile | Vehicles | Success | Pricing bias | Pricing MAE | Queue | All-late cloud | Inner iterations | Outer iterations | Outer residual | Wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['vehicle_count']} | "
            f"{100 * row['success_rate_mean']:.2f}% | "
            f"{100 * row['pricing_weighted_bias_mean']:+.2f} pp | "
            f"{100 * row['pricing_weighted_mae_mean']:.2f} pp | "
            f"{row['avg_cloud_queue_length_mean']:.2f} | "
            f"{100 * row['all_late_cloud_admission_rate_mean']:.2f}% | "
            f"{row['response_iterations_mean_mean']:.2f} | "
            f"{row['outer_iterations_mean_mean']:.2f} | "
            f"{max(row['outer_request_residual_mean_mean'], row['outer_cycle_residual_mean_mean']):.4f} | "
            f"{row['wall_clock_s_mean']:.1f}s |"
        )
    lines.extend(
        [
            "",
            f"Selected: **{selection['name']}** — {selection['reason']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _discover_checkpoint(root: Path, vehicles: int) -> Path:
    candidates = sorted(
        root.glob(
            f"run-*/training/hybrid_stackelberg-{vehicles}/*/dqn-policy.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Hybrid checkpoint not found below {root} for {vehicles} vehicles"
        )
    return candidates[0].resolve()


def _prepare_shared_inputs(pending) -> None:
    scenarios = {}
    caches = set()
    for _index, _profile, config in pending:
        if config.mobility != "sumo":
            continue
        if config.scenario_net:
            key = (
                config.scenario_net,
                config.route_output_dir,
                config.vehicle_count,
                config.route_departure_end_s,
                config.seed,
            )
            scenario = scenarios.get(key)
            if scenario is None:
                scenario = str(
                    prepare_sumo_scenario(
                        config.scenario_net,
                        config.route_output_dir,
                        config.vehicle_count,
                        config.route_departure_end_s,
                        config.seed,
                    )
                )
                scenarios[key] = scenario
            config.scenario_config = scenario
            config.scenario_net = None
        mobility = create_mobility(config)
        if not isinstance(mobility, TraceCachingMobilityProvider):
            continue
        cache = mobility.cache_path.resolve()
        if cache in caches or mobility.cache_is_valid():
            caches.add(cache)
            continue
        print(f"PREPARE MOBILITY vehicles={config.vehicle_count} seed={config.seed}")
        mobility.start()
        try:
            for step in range(config.steps):
                mobility.step(step)
        finally:
            mobility.close()
        caches.add(cache)


def _case_key(profile: str, config: SimulationConfig) -> str:
    return f"{profile}:v{config.vehicle_count}:seed{config.seed}:s{config.steps}"


def _signature(
    config_path: Path,
    checkpoints: dict[int, Path],
    profile_paths: list[Path],
) -> str:
    digest = sha256(config_path.read_bytes())
    for path in profile_paths:
        digest.update(
            json.dumps(
                SimulationConfig.from_toml(path).to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    for vehicles, path in sorted(checkpoints.items()):
        digest.update(str(vehicles).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_state(path: Path, signature: str) -> dict:
    if not path.exists():
        return {"signature": signature, "runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("signature") != signature:
        raise RuntimeError(f"existing ablation state has a different signature: {path}")
    return state


def _completed(run_dir: Path) -> bool:
    return (run_dir / "summary.json").is_file() and (run_dir / "timing.json").is_file()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
