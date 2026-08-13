from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import TraceCachingMobilityProvider, create_mobility
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


LEARNED_STRATEGIES = {"dqn"}
SUPPORTED_STRATEGIES = {"random", "greedy", "dqn", "stackelberg"}
KEY_METRICS = (
    "success_rate",
    "oracle_success_rate",
    "avg_success_latency_s",
    "avg_energy_j",
    "avg_cost_per_task",
    "avg_reward",
    "local_offload_ratio",
    "v2v_offload_ratio",
    "v2i_offload_ratio",
    "dqn_decision_ratio",
    "hybrid_deviation_ratio",
    "avg_cloud_queue_length",
    "max_cloud_queue_length",
    "avg_predicted_cloud_capacity_ratio",
    "all_actions_late_rate",
    "avoidable_failure_rate",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen baselines on the selected Hybrid diagnostic seed"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline_path = args.config.resolve()
    with pipeline_path.open("rb") as handle:
        pipeline = tomllib.load(handle)["pipeline"]
    _validate_pipeline(pipeline)
    repository = pipeline_path.parent.parent
    base = SimulationConfig.from_toml(
        (pipeline_path.parent / pipeline["base_config"]).resolve()
    )
    output = (repository / pipeline["output_dir"]).resolve()
    parallelism = int(args.parallelism or pipeline["parallelism"])
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    checkpoints = _find_dqn_checkpoints(base, pipeline, repository)
    configurations = _build_configurations(base, pipeline, output, checkpoints)

    if args.dry_run:
        for _key, config, _metadata in configurations:
            config.validate()
        hybrid_count = (
            len(pipeline["vehicle_counts"])
            if pipeline.get("include_hybrid_results", True)
            else 0
        )
        print(
            f"DRY RUN OK: {len(configurations)} baseline evaluations + "
            f"{hybrid_count} existing Hybrid rows, parallelism={parallelism}"
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    _check_free_disk(output, float(pipeline["minimum_free_disk_gb"]))
    state_path = output / "pipeline-state.json"
    state = _load_json(state_path, {"runs": {}})
    state["config"] = str(pipeline_path)
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)

    rows: list[dict] = []
    pending: list[tuple[str, SimulationConfig, dict]] = []
    for key, config, metadata in configurations:
        saved = state.get("runs", {}).get(key)
        if saved and _run_artifacts_exist(saved):
            rows.append(saved)
            print(
                f"SKIP {metadata['index']}/{metadata['total']} "
                f"{metadata['strategy']} vehicles={metadata['vehicles']}",
                flush=True,
            )
            continue
        recovered = _recover_completed(config)
        if recovered is not None:
            recovered.update(metadata)
            archive = _compress_task_records(Path(recovered["run_dir"]))
            recovered["task_record_archive"] = str(archive) if archive else ""
            state.setdefault("runs", {})[key] = recovered
            _write_json_atomic(state_path, state)
            rows.append(recovered)
            print(
                f"RECOVER {metadata['index']}/{metadata['total']} "
                f"{metadata['strategy']} vehicles={metadata['vehicles']}",
                flush=True,
            )
            continue
        pending.append((key, config, metadata))

    _prepare_shared_mobility([case[1] for case in pending])
    for key, metadata, row in _execute_cases(pending, parallelism):
        row.update(metadata)
        archive = _compress_task_records(Path(row["run_dir"]))
        row["task_record_archive"] = str(archive) if archive else ""
        state.setdefault("runs", {})[key] = row
        _write_json_atomic(state_path, state)
        rows.append(row)
        print(
            f"DONE {metadata['index']}/{metadata['total']} "
            f"{metadata['strategy']} vehicles={metadata['vehicles']} "
            f"success={100 * row['success_rate']:.2f}% "
            f"wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )

    comparison_rows = list(rows)
    if pipeline.get("include_hybrid_results", True):
        comparison_rows.extend(
            _read_hybrid_rows(repository, pipeline, int(pipeline["evaluation_seed"]))
        )
    comparison_rows.sort(
        key=lambda row: (
            int(row["configured_vehicle_count"]),
            _strategy_order(row["strategy"]),
        )
    )
    _write_csv(output / "baseline-results.csv", rows)
    _write_csv(output / "comparison-results.csv", comparison_rows)
    key_rows = _key_metric_rows(comparison_rows)
    _write_csv(output / "comparison-key-metrics.csv", key_rows)
    (output / "comparison-summary.md").write_text(
        _render_summary(pipeline, key_rows),
        encoding="utf-8",
    )
    storage = _storage_manifest(output, rows, pipeline)
    _write_json_atomic(output / "storage-manifest.json", storage)
    state["complete"] = True
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    print(f"COMPLETE {output}")
    print(f"SUMMARY {output / 'comparison-summary.md'}")
    return 0


def _build_configurations(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
    checkpoints: dict[int, Path],
) -> list[tuple[str, SimulationConfig, dict]]:
    cases = []
    total = len(pipeline["vehicle_counts"]) * len(pipeline["strategies"])
    index = 0
    for vehicles in pipeline["vehicle_counts"]:
        for strategy in pipeline["strategies"]:
            index += 1
            strategy = str(strategy)
            vehicles = int(vehicles)
            config = copy.deepcopy(base)
            config.strategy = strategy
            config.steps = int(pipeline["steps"])
            config.vehicle_count = vehicles
            config.seed = int(pipeline["evaluation_seed"])
            config.output_dir = str(
                output / "runs" / f"{strategy}-v{vehicles}-seed{config.seed}"
            )
            config.record_decision_diagnostics = True
            config.record_task_records = True
            config.task_record_sample_rate = float(pipeline["task_sample_rate"])
            config.minimum_free_disk_gb = float(pipeline["minimum_free_disk_gb"])
            if strategy in LEARNED_STRATEGIES:
                config.dqn.mode = "evaluate"
                config.dqn.checkpoint_path = str(checkpoints[vehicles])
            else:
                config.dqn.mode = "train"
                config.dqn.checkpoint_path = None
            config.validate()
            metadata = {
                "index": index,
                "total": total,
                "strategy": strategy,
                "vehicles": vehicles,
                "evaluation_seed": config.seed,
                "detail_kind": "sampled",
            }
            cases.append(
                (
                    f"{strategy}:v{vehicles}:seed{config.seed}",
                    config,
                    metadata,
                )
            )
    return cases


def _find_dqn_checkpoints(
    base: SimulationConfig,
    pipeline: dict,
    repository: Path,
) -> dict[int, Path]:
    checkpoints = {}
    expected_seed = int(pipeline["dqn_training_seed"])
    for vehicles in pipeline["vehicle_counts"]:
        matches = []
        for root_name in pipeline["dqn_checkpoint_roots"]:
            root = (repository / root_name).resolve()
            if not root.exists():
                continue
            for config_path in root.rglob("config.json"):
                try:
                    value = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    value.get("strategy") == "dqn"
                    and value.get("cloud_pricing_mode") == base.cloud_pricing_mode
                    and int(value.get("seed", -1)) == expected_seed
                    and int(value.get("vehicle_count", -1)) == int(vehicles)
                    and value.get("dqn", {}).get("mode") == "train"
                    and tuple(value.get("dqn", {}).get("hidden_sizes", ()))
                    == base.dqn.hidden_sizes
                ):
                    checkpoint = config_path.parent / "dqn-policy.pt"
                    if checkpoint.exists():
                        matches.append(checkpoint)
        if not matches:
            raise FileNotFoundError(
                f"no compatible DQN checkpoint for vehicles={vehicles}, "
                f"training_seed={expected_seed}"
            )
        checkpoints[int(vehicles)] = sorted(
            matches,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[0]
    return checkpoints


def _execute_cases(
    cases: list[tuple[str, SimulationConfig, dict]],
    parallelism: int,
):
    if not cases:
        return
    ordered = sorted(
        cases,
        key=lambda case: (case[1].vehicle_count, case[1].strategy),
        reverse=True,
    )
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        futures = {}
        for key, config, metadata in ordered:
            print(
                f"START {metadata['index']}/{metadata['total']} "
                f"{metadata['strategy']} vehicles={metadata['vehicles']}",
                flush=True,
            )
            futures[executor.submit(_execute, config)] = (key, metadata)
        failures = []
        for future in as_completed(futures):
            key, metadata = futures[future]
            try:
                yield key, metadata, future.result()
            except Exception as error:
                failures.append((metadata, error))
                print(
                    f"FAILED {metadata['strategy']} "
                    f"vehicles={metadata['vehicles']}: {error}",
                    flush=True,
                )
        if failures:
            metadata, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} baseline case(s) failed; first failure was "
                f"{metadata['strategy']} vehicles={metadata['vehicles']}"
            ) from error


def _execute(config: SimulationConfig) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    return {
        "phase": "evaluate",
        **asdict(summary),
        "wall_clock_s": perf_counter() - wall_started,
        "process_cpu_s": process_time() - cpu_started,
        "run_dir": run_dir,
    }


def _read_hybrid_rows(
    repository: Path,
    pipeline: dict,
    expected_seed: int,
) -> list[dict]:
    source = (repository / pipeline["hybrid_results_csv"]).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Hybrid result CSV does not exist: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if int(row["seed"]) == expected_seed
        and int(row["configured_vehicle_count"]) in {
            int(value) for value in pipeline["vehicle_counts"]
        }
    ]
    if len(selected) != len(pipeline["vehicle_counts"]):
        raise RuntimeError(
            f"expected {len(pipeline['vehicle_counts'])} Hybrid rows for "
            f"seed={expected_seed}, received {len(selected)}"
        )
    for row in selected:
        row["source"] = str(source)
    return selected


def _key_metric_rows(rows: list[dict]) -> list[dict]:
    hybrid_by_vehicle = {
        int(row["configured_vehicle_count"]): float(row["success_rate"])
        for row in rows
        if row["strategy"] == "hybrid_stackelberg"
    }
    values = []
    for row in rows:
        vehicles = int(row["configured_vehicle_count"])
        result = {
            "vehicle_count": vehicles,
            "strategy": row["strategy"],
            "seed": int(row["seed"]),
            "total_tasks": int(row["total_tasks"]),
        }
        for metric in KEY_METRICS:
            value = row.get(metric, "")
            result[metric] = float(value) if value not in ("", None) else ""
        hybrid_success = hybrid_by_vehicle.get(vehicles)
        result["hybrid_advantage_pp"] = (
            100.0 * (hybrid_success - float(row["success_rate"]))
            if hybrid_success is not None
            else ""
        )
        result["wall_clock_s"] = (
            float(row["wall_clock_s"]) if row.get("wall_clock_s") else ""
        )
        result["run_dir"] = row.get("run_dir", "")
        result["task_record_archive"] = row.get("task_record_archive", "")
        values.append(result)
    return sorted(
        values,
        key=lambda row: (
            row["vehicle_count"],
            -float(row["success_rate"]),
            _strategy_order(row["strategy"]),
        ),
    )


def _render_summary(pipeline: dict, rows: list[dict]) -> str:
    lines = [
        "# Seed 48 strategy comparison",
        "",
        f"- Evaluation seed: {pipeline['evaluation_seed']}",
        f"- Steps: {pipeline['steps']}",
        f"- Frozen DQN training seed: {pipeline['dqn_training_seed']}",
        (
            "- Baseline task-record sample: "
            f"{100 * float(pipeline['task_sample_rate']):.3f}%"
        ),
        "- All aggregate metrics are exact; only task-level rows are sampled.",
        "",
        "| Vehicles | Strategy | Success | Hybrid advantage | "
        "Success latency (s) | Energy/task (J) | Cost/task | Oracle |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap = row["hybrid_advantage_pp"]
        gap_text = f"{gap:+.2f} pp" if gap != "" else ""
        lines.append(
            f"| {row['vehicle_count']} | {row['strategy']} | "
            f"{100 * row['success_rate']:.2f}% | "
            f"{gap_text} | "
            f"{row['avg_success_latency_s']:.3f} | "
            f"{row['avg_energy_j']:.2f} | "
            f"{row['avg_cost_per_task']:.3f} | "
            f"{100 * row['oracle_success_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Storage policy",
            "",
            "- `summary.json`, `timing.json`, and `pricing.jsonl` are retained.",
            "- A deterministic task sample is compressed to `tasks.csv.gz`.",
            "- No training runs or full baseline task tables are created.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_shared_mobility(configurations: list[SimulationConfig]) -> None:
    prepared_scenarios: dict[tuple[str, str, int, float, int], str] = {}
    prepared_caches: set[Path] = set()
    for config in configurations:
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
            scenario = prepared_scenarios.get(key)
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
                prepared_scenarios[key] = scenario
            config.scenario_config = scenario
            config.scenario_net = None
        mobility = create_mobility(config)
        if not isinstance(mobility, TraceCachingMobilityProvider):
            mobility.close()
            continue
        cache_path = mobility.cache_path.resolve()
        if cache_path in prepared_caches or mobility.cache_is_valid():
            prepared_caches.add(cache_path)
            mobility.close()
            continue
        print(
            f"PREPARE MOBILITY vehicles={config.vehicle_count} seed={config.seed}",
            flush=True,
        )
        mobility.start()
        try:
            for step in range(config.steps):
                mobility.step(step)
        finally:
            mobility.close()
        prepared_caches.add(cache_path)


def _recover_completed(config: SimulationConfig) -> dict | None:
    output = Path(config.output_dir)
    if not output.exists():
        return None
    for run_dir in sorted(
        (path for path in output.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        summary_path = run_dir / "summary.json"
        timing_path = run_dir / "timing.json"
        if not summary_path.exists() or not timing_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("strategy") != config.strategy
            or int(summary.get("configured_vehicle_count", -1))
            != config.vehicle_count
            or int(summary.get("configured_steps", -1)) != config.steps
            or int(summary.get("completed_steps", -1)) != config.steps
            or int(summary.get("seed", -1)) != config.seed
        ):
            continue
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        return {
            "phase": "evaluate",
            **summary,
            "wall_clock_s": float(timing["wall_clock_s"]),
            "process_cpu_s": 0.0,
            "run_dir": str(run_dir),
        }
    return None


def _compress_task_records(run_dir: Path) -> Path | None:
    source = run_dir / "tasks.csv"
    archive = run_dir / "tasks.csv.gz"
    if not source.exists():
        return archive if archive.exists() else None
    temporary = run_dir / "tasks.csv.gz.tmp"
    with source.open("rb") as source_handle, gzip.open(
        temporary,
        "wb",
        compresslevel=6,
    ) as archive_handle:
        shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)
    temporary.replace(archive)
    source.unlink()
    return archive


def _storage_manifest(output: Path, rows: list[dict], pipeline: dict) -> dict:
    records_seen = 0
    records_written = 0
    for row in rows:
        path = Path(row["run_dir"]) / "task-recording.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        records_seen += int(value.get("records_seen", 0))
        records_written += int(value.get("records_written", 0))
    files = [path for path in output.rglob("*") if path.is_file()]
    return {
        "task_sample_rate_configured": float(pipeline["task_sample_rate"]),
        "records_seen": records_seen,
        "records_written": records_written,
        "task_sample_rate_actual": (
            records_written / records_seen if records_seen else 0.0
        ),
        "result_files": len(files),
        "result_size_mib": sum(path.stat().st_size for path in files) / 1024**2,
        "free_disk_gib": shutil.disk_usage(output).free / 1024**3,
    }


def _run_artifacts_exist(row: dict) -> bool:
    run_dir = Path(row["run_dir"])
    return (
        (run_dir / "summary.json").exists()
        and (run_dir / "timing.json").exists()
        and (
            (run_dir / "tasks.csv.gz").exists()
            or (run_dir / "tasks.csv").exists()
        )
    )


def _validate_pipeline(pipeline: dict) -> None:
    required = {
        "base_config",
        "output_dir",
        "steps",
        "vehicle_counts",
        "strategies",
        "evaluation_seed",
        "dqn_training_seed",
        "parallelism",
        "task_sample_rate",
        "minimum_free_disk_gb",
        "dqn_checkpoint_roots",
    }
    missing = required - set(pipeline)
    if missing:
        raise ValueError(f"missing pipeline settings: {sorted(missing)}")
    if int(pipeline["steps"]) <= 0:
        raise ValueError("steps must be positive")
    if not pipeline["vehicle_counts"]:
        raise ValueError("vehicle_counts must not be empty")
    strategies = {str(value) for value in pipeline["strategies"]}
    unknown = strategies - SUPPORTED_STRATEGIES
    if unknown:
        raise ValueError(f"unsupported baseline strategies: {sorted(unknown)}")
    if strategies != SUPPORTED_STRATEGIES:
        raise ValueError(
            "strategies must contain random, greedy, dqn, and stackelberg"
        )
    if not 0.0 < float(pipeline["task_sample_rate"]) <= 1.0:
        raise ValueError("task_sample_rate must be in (0, 1]")
    if float(pipeline["minimum_free_disk_gb"]) < 0.0:
        raise ValueError("minimum_free_disk_gb must be non-negative")


def _strategy_order(strategy: str) -> int:
    return {
        "random": 0,
        "greedy": 1,
        "dqn": 2,
        "stackelberg": 3,
        "hybrid_stackelberg": 4,
    }.get(strategy, 99)


def _check_free_disk(path: Path, minimum_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB is free; "
            f"minimum_free_disk_gb is {minimum_gb:.2f}"
        )


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
