from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import (
    TraceCachingMobilityProvider,
    create_mobility,
)
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Hybrid checkpoints across seeds and validate them independently"
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

    planned_training = len(pipeline["training_seeds"]) * len(pipeline["vehicle_counts"])
    planned_validation = (
        planned_training * len(pipeline["validation_seeds"])
    )
    planned_diagnostics = len(pipeline["vehicle_counts"])
    if args.dry_run:
        _validate_configurations(base, pipeline, output)
        print(
            "DRY RUN OK: "
            f"{planned_training} checkpoint slots, "
            f"{planned_validation} sampled validations, "
            f"{planned_diagnostics} selected diagnostics, "
            f"parallelism={parallelism}"
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "pipeline-state.json"
    state = _load_json(state_path, {"runs": {}, "checkpoints": {}})
    state["config"] = str(pipeline_path)
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    _check_free_disk(output, float(pipeline["minimum_free_disk_gb"]))

    checkpoints = _prepare_checkpoints(
        base,
        pipeline,
        repository,
        output,
        state,
        state_path,
        parallelism,
    )
    validation_rows = _run_validations(
        base,
        pipeline,
        output,
        checkpoints,
        state,
        state_path,
        parallelism,
    )
    selection_rows = _select_checkpoints(validation_rows, pipeline, checkpoints)
    _write_csv(output / "validation-results.csv", validation_rows)
    _write_csv(output / "checkpoint-selection.csv", selection_rows)

    diagnostic_rows, storage = _run_selected_diagnostics(
        base,
        pipeline,
        output,
        selection_rows,
        state,
        state_path,
        parallelism,
        validation_rows,
    )
    _write_csv(output / "selected-diagnostics.csv", diagnostic_rows)
    _write_json_atomic(output / "storage-plan.json", storage)
    (output / "hybrid-seed-summary.md").write_text(
        _render_summary(
            pipeline,
            validation_rows,
            selection_rows,
            diagnostic_rows,
            storage,
        ),
        encoding="utf-8",
    )
    state["complete"] = True
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    print(f"COMPLETE {output}")
    print(f"SUMMARY {output / 'hybrid-seed-summary.md'}")
    return 0


def _prepare_checkpoints(
    base: SimulationConfig,
    pipeline: dict,
    repository: Path,
    output: Path,
    state: dict,
    state_path: Path,
    parallelism: int,
) -> dict[tuple[int, int], Path]:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[tuple[int, int], Path] = {}
    pending: list[tuple[str, SimulationConfig, dict]] = []
    total = len(pipeline["training_seeds"]) * len(pipeline["vehicle_counts"])
    index = 0
    for training_seed in pipeline["training_seeds"]:
        for vehicles in pipeline["vehicle_counts"]:
            index += 1
            key = _case_key("train", vehicles, training_seed)
            destination = checkpoint_dir / f"hybrid-v{vehicles}-seed{training_seed}.pt"
            if destination.exists():
                checkpoints[(int(training_seed), int(vehicles))] = destination
                print(
                    f"SKIP TRAIN {index}/{total} seed={training_seed} vehicles={vehicles}",
                    flush=True,
                )
                continue
            reused = _find_compatible_checkpoint(
                base,
                pipeline,
                repository,
                int(training_seed),
                int(vehicles),
            )
            if reused is not None:
                shutil.copy2(reused, destination)
                source_record = {
                    "source": str(reused),
                    "copied_to": str(destination),
                    "training_seed": int(training_seed),
                    "vehicles": int(vehicles),
                    "reused": True,
                }
                _write_json_atomic(destination.with_suffix(".source.json"), source_record)
                state.setdefault("checkpoints", {})[key] = source_record
                _write_json_atomic(state_path, state)
                checkpoints[(int(training_seed), int(vehicles))] = destination
                print(
                    f"REUSE TRAIN {index}/{total} seed={training_seed} "
                    f"vehicles={vehicles}",
                    flush=True,
                )
                continue
            config = _training_config(
                base,
                pipeline,
                output,
                int(training_seed),
                int(vehicles),
            )
            pending.append(
                (
                    key,
                    config,
                    {
                        "index": index,
                        "total": total,
                        "training_seed": int(training_seed),
                        "vehicles": int(vehicles),
                        "destination": str(destination),
                    },
                )
            )

    for key, metadata, row in _execute_cases(pending, "TRAIN", parallelism):
        source = Path(row["run_dir"]) / "dqn-policy.pt"
        destination = Path(metadata["destination"])
        if not source.exists():
            raise RuntimeError(f"training checkpoint was not created: {source}")
        shutil.copy2(source, destination)
        record = {
            "source": str(source),
            "copied_to": str(destination),
            "training_seed": metadata["training_seed"],
            "vehicles": metadata["vehicles"],
            "reused": False,
            "row": row,
        }
        _write_json_atomic(destination.with_suffix(".source.json"), record)
        state.setdefault("checkpoints", {})[key] = record
        state.setdefault("runs", {})[key] = row
        _write_json_atomic(state_path, state)
        checkpoints[(metadata["training_seed"], metadata["vehicles"])] = destination
        print(
            f"DONE TRAIN {metadata['index']}/{metadata['total']} "
            f"seed={metadata['training_seed']} vehicles={metadata['vehicles']} "
            f"success={100 * row['success_rate']:.2f}% "
            f"wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )
    return checkpoints


def _run_validations(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
    checkpoints: dict[tuple[int, int], Path],
    state: dict,
    state_path: Path,
    parallelism: int,
) -> list[dict]:
    cases: list[tuple[str, SimulationConfig, dict]] = []
    rows: list[dict] = []
    total = (
        len(pipeline["training_seeds"])
        * len(pipeline["validation_seeds"])
        * len(pipeline["vehicle_counts"])
    )
    index = 0
    for training_seed in pipeline["training_seeds"]:
        for validation_seed in pipeline["validation_seeds"]:
            for vehicles in pipeline["vehicle_counts"]:
                index += 1
                key = _case_key(
                    "validate",
                    vehicles,
                    training_seed,
                    validation_seed,
                )
                saved = state.get("runs", {}).get(key)
                if saved and Path(saved["run_dir"], "summary.json").exists():
                    rows.append(saved)
                    print(
                        f"SKIP VALIDATE {index}/{total} train={training_seed} "
                        f"eval={validation_seed} vehicles={vehicles}",
                        flush=True,
                    )
                    continue
                config = _evaluation_config(
                    base,
                    pipeline,
                    output,
                    checkpoints[(int(training_seed), int(vehicles))],
                    int(training_seed),
                    int(validation_seed),
                    int(vehicles),
                    detail_kind="sampled",
                    sample_rate=float(pipeline["validation_task_sample_rate"]),
                )
                recovered = _recover_completed(config, "validate")
                if recovered is not None:
                    recovered.update(
                        training_seed=int(training_seed),
                        evaluation_seed=int(validation_seed),
                        detail_kind="sampled",
                    )
                    state.setdefault("runs", {})[key] = recovered
                    _write_json_atomic(state_path, state)
                    rows.append(recovered)
                    print(
                        f"RECOVER VALIDATE {index}/{total} train={training_seed} "
                        f"eval={validation_seed} vehicles={vehicles}",
                        flush=True,
                    )
                    continue
                cases.append(
                    (
                        key,
                        config,
                        {
                            "index": index,
                            "total": total,
                            "training_seed": int(training_seed),
                            "evaluation_seed": int(validation_seed),
                            "vehicles": int(vehicles),
                            "detail_kind": "sampled",
                        },
                    )
                )
    _prepare_shared_mobility([case[1] for case in cases])
    for key, metadata, row in _execute_cases(cases, "VALIDATE", parallelism):
        row.update(
            training_seed=metadata["training_seed"],
            evaluation_seed=metadata["evaluation_seed"],
            detail_kind=metadata["detail_kind"],
        )
        state.setdefault("runs", {})[key] = row
        _write_json_atomic(state_path, state)
        rows.append(row)
        print(
            f"DONE VALIDATE {metadata['index']}/{metadata['total']} "
            f"train={metadata['training_seed']} eval={metadata['evaluation_seed']} "
            f"vehicles={metadata['vehicles']} "
            f"success={100 * row['success_rate']:.2f}% "
            f"wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["configured_vehicle_count"]),
            int(row["training_seed"]),
            int(row["evaluation_seed"]),
        ),
    )


def _select_checkpoints(
    validation_rows: list[dict],
    pipeline: dict,
    checkpoints: dict[tuple[int, int], Path],
) -> list[dict]:
    selected = []
    penalty = float(pipeline["stability_penalty"])
    for vehicles in pipeline["vehicle_counts"]:
        candidates = []
        for training_seed in pipeline["training_seeds"]:
            rows = [
                row
                for row in validation_rows
                if int(row["configured_vehicle_count"]) == int(vehicles)
                and int(row["training_seed"]) == int(training_seed)
            ]
            if len(rows) != len(pipeline["validation_seeds"]):
                raise RuntimeError(
                    f"incomplete validation set for seed={training_seed}, "
                    f"vehicles={vehicles}"
                )
            success = [float(row["success_rate"]) for row in rows]
            average = mean(success)
            sample_std = stdev(success) if len(success) > 1 else 0.0
            candidates.append(
                {
                    "vehicle_count": int(vehicles),
                    "training_seed": int(training_seed),
                    "validation_runs": len(rows),
                    "mean_success_rate": average,
                    "sample_std": sample_std,
                    "min_success_rate": min(success),
                    "max_success_rate": max(success),
                    "stability_score": average - penalty * sample_std,
                    "mean_reward": mean(float(row["avg_reward"]) for row in rows),
                    "mean_hybrid_deviation_ratio": mean(
                        float(row["hybrid_deviation_ratio"]) for row in rows
                    ),
                    "mean_beneficial_deviation_rate": mean(
                        float(row["hybrid_beneficial_deviation_rate"])
                        for row in rows
                    ),
                    "checkpoint": str(
                        checkpoints[(int(training_seed), int(vehicles))]
                    ),
                }
            )
        winner = max(
            candidates,
            key=lambda row: (
                row["stability_score"],
                row["mean_success_rate"],
                row["min_success_rate"],
                -row["training_seed"],
            ),
        )
        for row in candidates:
            row["selected"] = row is winner
            selected.append(row)
    return sorted(
        selected,
        key=lambda row: (row["vehicle_count"], -int(row["selected"]), row["training_seed"]),
    )


def _run_selected_diagnostics(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
    selection_rows: list[dict],
    state: dict,
    state_path: Path,
    parallelism: int,
    validation_rows: list[dict],
) -> tuple[list[dict], dict]:
    winners = [row for row in selection_rows if row["selected"]]
    bytes_per_record = _estimate_bytes_per_record(validation_rows)
    estimated_raw_bytes = sum(
        _mean_total_tasks(
            validation_rows,
            int(row["vehicle_count"]),
            int(row["training_seed"]),
        )
        * bytes_per_record
        for row in winners
    )
    drive = shutil.disk_usage(output)
    reserve_bytes = int(float(pipeline["minimum_free_disk_gb"]) * 1024**3)
    configured_budget = int(float(pipeline["diagnostic_detail_budget_gb"]) * 1024**3)
    available_budget = max(0, drive.free - reserve_bytes - 1024**3)
    raw_budget = min(configured_budget, available_budget)
    sample_rate = (
        min(1.0, raw_budget / max(estimated_raw_bytes, 1))
        if raw_budget > 0
        else 0.0
    )
    storage = {
        "free_before_diagnostics_gb": drive.free / 1024**3,
        "minimum_free_disk_gb": float(pipeline["minimum_free_disk_gb"]),
        "diagnostic_detail_budget_gb": float(
            pipeline["diagnostic_detail_budget_gb"]
        ),
        "estimated_bytes_per_task_record": bytes_per_record,
        "estimated_selected_raw_gb": estimated_raw_bytes / 1024**3,
        "diagnostic_task_sample_rate": sample_rate,
        "compression": "gzip",
    }
    if sample_rate <= 0.0:
        storage["diagnostics_skipped"] = "insufficient free disk above reserve"
        return [], storage

    cases: list[tuple[str, SimulationConfig, dict]] = []
    rows: list[dict] = []
    diagnostic_seed = int(pipeline["diagnostic_seed"])
    for index, winner in enumerate(winners, start=1):
        vehicles = int(winner["vehicle_count"])
        training_seed = int(winner["training_seed"])
        key = _case_key("diagnostic", vehicles, training_seed, diagnostic_seed)
        saved = state.get("runs", {}).get(key)
        if saved and (
            Path(saved["run_dir"], "tasks.csv").exists()
            or Path(saved["run_dir"], "tasks.csv.gz").exists()
        ):
            rows.append(saved)
            print(
                f"SKIP DIAGNOSTIC {index}/{len(winners)} train={training_seed} "
                f"eval={diagnostic_seed} vehicles={vehicles}",
                flush=True,
            )
            continue
        config = _evaluation_config(
            base,
            pipeline,
            output,
            Path(winner["checkpoint"]),
            training_seed,
            diagnostic_seed,
            vehicles,
            detail_kind="selected",
            sample_rate=sample_rate,
        )
        cases.append(
            (
                key,
                config,
                {
                    "index": index,
                    "total": len(winners),
                    "training_seed": training_seed,
                    "evaluation_seed": diagnostic_seed,
                    "vehicles": vehicles,
                    "detail_kind": "selected",
                },
            )
        )
    _prepare_shared_mobility([case[1] for case in cases])
    for key, metadata, row in _execute_cases(cases, "DIAGNOSTIC", parallelism):
        row.update(
            training_seed=metadata["training_seed"],
            evaluation_seed=metadata["evaluation_seed"],
            detail_kind=metadata["detail_kind"],
        )
        compressed = _compress_task_records(Path(row["run_dir"]))
        row["task_record_archive"] = str(compressed) if compressed else ""
        state.setdefault("runs", {})[key] = row
        _write_json_atomic(state_path, state)
        rows.append(row)
        print(
            f"DONE DIAGNOSTIC {metadata['index']}/{metadata['total']} "
            f"train={metadata['training_seed']} eval={metadata['evaluation_seed']} "
            f"vehicles={metadata['vehicles']} "
            f"success={100 * row['success_rate']:.2f}% "
            f"wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )
    storage["free_after_diagnostics_gb"] = shutil.disk_usage(output).free / 1024**3
    return sorted(rows, key=lambda row: int(row["configured_vehicle_count"])), storage


def _training_config(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
    seed: int,
    vehicles: int,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = "hybrid_stackelberg"
    config.steps = int(pipeline["training_steps"])
    config.vehicle_count = vehicles
    config.seed = seed
    config.output_dir = str(output / "runs" / "training" / f"v{vehicles}-seed{seed}")
    config.dqn.mode = "train"
    config.dqn.checkpoint_path = None
    config.record_decision_diagnostics = False
    config.record_task_records = False
    config.task_record_sample_rate = 0.0
    config.minimum_free_disk_gb = float(pipeline["minimum_free_disk_gb"])
    config.validate()
    return config


def _evaluation_config(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
    checkpoint: Path,
    training_seed: int,
    evaluation_seed: int,
    vehicles: int,
    *,
    detail_kind: str,
    sample_rate: float,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = "hybrid_stackelberg"
    config.steps = int(pipeline["evaluation_steps"])
    config.vehicle_count = vehicles
    config.seed = evaluation_seed
    config.output_dir = str(
        output
        / "runs"
        / detail_kind
        / f"v{vehicles}-train{training_seed}-eval{evaluation_seed}"
    )
    config.dqn.mode = "evaluate"
    config.dqn.checkpoint_path = str(checkpoint)
    config.record_decision_diagnostics = True
    config.record_task_records = sample_rate > 0.0
    config.task_record_sample_rate = sample_rate
    config.minimum_free_disk_gb = float(pipeline["minimum_free_disk_gb"])
    config.validate()
    return config


def _execute_cases(
    cases: list[tuple[str, SimulationConfig, dict]],
    label: str,
    parallelism: int,
):
    if not cases:
        return
    ordered = sorted(
        cases,
        key=lambda case: (
            case[1].vehicle_count,
            case[1].steps,
        ),
        reverse=True,
    )
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        futures = {}
        for key, config, metadata in ordered:
            print(
                f"START {label} {metadata['index']}/{metadata['total']} "
                f"train={metadata.get('training_seed')} "
                f"eval={metadata.get('evaluation_seed', '-')} "
                f"vehicles={metadata['vehicles']}",
                flush=True,
            )
            futures[executor.submit(_execute, config, label.lower())] = (
                key,
                metadata,
            )
        failures = []
        for future in as_completed(futures):
            key, metadata = futures[future]
            try:
                yield key, metadata, future.result()
            except Exception as error:
                failures.append((metadata, error))
                print(
                    f"FAILED {label} train={metadata.get('training_seed')} "
                    f"eval={metadata.get('evaluation_seed', '-')} "
                    f"vehicles={metadata['vehicles']}: {error}",
                    flush=True,
                )
        if failures:
            metadata, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} {label.lower()} case(s) failed; first failure "
                f"was train={metadata.get('training_seed')} "
                f"eval={metadata.get('evaluation_seed', '-')} "
                f"vehicles={metadata['vehicles']}"
            ) from error


def _execute(config: SimulationConfig, phase: str) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    return {
        "phase": phase,
        **asdict(summary),
        "wall_clock_s": perf_counter() - wall_started,
        "process_cpu_s": process_time() - cpu_started,
        "run_dir": run_dir,
    }


def _find_compatible_checkpoint(
    base: SimulationConfig,
    pipeline: dict,
    repository: Path,
    training_seed: int,
    vehicles: int,
) -> Path | None:
    expected = _training_config(
        base,
        pipeline,
        repository / pipeline["output_dir"],
        training_seed,
        vehicles,
    ).to_dict()
    expected_signature = _compatibility_signature(expected)
    for root_name in pipeline.get("reuse_checkpoint_roots", []):
        root = (repository / root_name).resolve()
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                int(existing.get("seed", -1)) != training_seed
                or int(existing.get("vehicle_count", -1)) != vehicles
                or int(existing.get("steps", -1)) != int(pipeline["training_steps"])
                or existing.get("strategy") != "hybrid_stackelberg"
                or existing.get("dqn", {}).get("mode") != "train"
            ):
                continue
            if _compatibility_signature(existing) != expected_signature:
                continue
            checkpoint = config_path.parent / "dqn-policy.pt"
            if checkpoint.exists():
                return checkpoint
    return None


def _compatibility_signature(value: dict) -> str:
    normalized = copy.deepcopy(value)
    for key in (
        "output_dir",
        "record_decision_diagnostics",
        "record_task_records",
        "task_record_sample_rate",
        "minimum_free_disk_gb",
        "scenario_config",
        "scenario_net",
        "route_output_dir",
        "sumo_binary",
    ):
        normalized.pop(key, None)
    normalized.get("dqn", {}).pop("checkpoint_path", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _recover_completed(config: SimulationConfig, phase: str) -> dict | None:
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
            "phase": phase,
            **summary,
            "wall_clock_s": float(timing["wall_clock_s"]),
            "process_cpu_s": 0.0,
            "run_dir": str(run_dir),
        }
    return None


def _prepare_shared_mobility(configurations: list[SimulationConfig]) -> None:
    if not configurations:
        return
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


def _estimate_bytes_per_record(rows: list[dict]) -> float:
    total_bytes = 0
    total_records = 0
    for row in rows:
        run_dir = Path(row["run_dir"])
        recording_path = run_dir / "task-recording.json"
        task_path = run_dir / "tasks.csv"
        if not recording_path.exists() or not task_path.exists():
            continue
        recording = json.loads(recording_path.read_text(encoding="utf-8"))
        written = int(recording.get("records_written", 0))
        if written <= 0:
            continue
        total_bytes += task_path.stat().st_size
        total_records += written
    return total_bytes / total_records if total_records else 900.0


def _mean_total_tasks(
    rows: list[dict],
    vehicles: int,
    training_seed: int,
) -> float:
    values = [
        int(row["total_tasks"])
        for row in rows
        if int(row["configured_vehicle_count"]) == vehicles
        and int(row["training_seed"]) == training_seed
    ]
    return mean(values) if values else 0.0


def _compress_task_records(run_dir: Path) -> Path | None:
    source = run_dir / "tasks.csv"
    if not source.exists():
        archive = run_dir / "tasks.csv.gz"
        return archive if archive.exists() else None
    archive = run_dir / "tasks.csv.gz"
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


def _validate_pipeline(pipeline: dict) -> None:
    required = {
        "base_config",
        "output_dir",
        "training_steps",
        "evaluation_steps",
        "vehicle_counts",
        "training_seeds",
        "validation_seeds",
        "diagnostic_seed",
        "parallelism",
        "validation_task_sample_rate",
        "diagnostic_detail_budget_gb",
        "minimum_free_disk_gb",
        "stability_penalty",
    }
    missing = required - set(pipeline)
    if missing:
        raise ValueError(f"missing pipeline settings: {sorted(missing)}")
    if min(int(pipeline["training_steps"]), int(pipeline["evaluation_steps"])) <= 0:
        raise ValueError("training and evaluation steps must be positive")
    for key in ("vehicle_counts", "training_seeds", "validation_seeds"):
        if not pipeline[key]:
            raise ValueError(f"{key} must not be empty")
    sample_rate = float(pipeline["validation_task_sample_rate"])
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("validation_task_sample_rate must be in (0, 1]")
    if float(pipeline["diagnostic_detail_budget_gb"]) <= 0.0:
        raise ValueError("diagnostic_detail_budget_gb must be positive")
    if float(pipeline["minimum_free_disk_gb"]) < 0.0:
        raise ValueError("minimum_free_disk_gb must be non-negative")
    if float(pipeline["stability_penalty"]) < 0.0:
        raise ValueError("stability_penalty must be non-negative")


def _validate_configurations(
    base: SimulationConfig,
    pipeline: dict,
    output: Path,
) -> None:
    checkpoint = output / "dry-run-policy.pt"
    for training_seed in pipeline["training_seeds"]:
        for vehicles in pipeline["vehicle_counts"]:
            _training_config(
                base,
                pipeline,
                output,
                int(training_seed),
                int(vehicles),
            ).validate()
            for evaluation_seed in pipeline["validation_seeds"]:
                _evaluation_config(
                    base,
                    pipeline,
                    output,
                    checkpoint,
                    int(training_seed),
                    int(evaluation_seed),
                    int(vehicles),
                    detail_kind="sampled",
                    sample_rate=float(pipeline["validation_task_sample_rate"]),
                ).validate()


def _render_summary(
    pipeline: dict,
    validation_rows: list[dict],
    selection_rows: list[dict],
    diagnostic_rows: list[dict],
    storage: dict,
) -> str:
    lines = [
        "# Hybrid seed stability",
        "",
        (
            f"- Training seeds: {', '.join(str(value) for value in pipeline['training_seeds'])}"
        ),
        (
            f"- Validation seeds: {', '.join(str(value) for value in pipeline['validation_seeds'])}"
        ),
        f"- Validation runs: {len(validation_rows)}",
        (
            "- Selection score: mean success rate - "
            f"{pipeline['stability_penalty']} × sample standard deviation"
        ),
        (
            "- Validation task detail sample: "
            f"{100 * float(pipeline['validation_task_sample_rate']):.3f}%"
        ),
        (
            "- Selected diagnostic task detail sample: "
            f"{100 * float(storage['diagnostic_task_sample_rate']):.2f}%"
        ),
        "",
        "## Checkpoint selection",
        "",
        "| Vehicles | Training seed | Mean success | Std | Minimum | Score | Selected |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['vehicle_count']} | {row['training_seed']} | "
            f"{100 * row['mean_success_rate']:.2f}% | "
            f"{100 * row['sample_std']:.2f} pp | "
            f"{100 * row['min_success_rate']:.2f}% | "
            f"{100 * row['stability_score']:.2f} | "
            f"{'yes' if row['selected'] else ''} |"
        )
    lines.extend(
        [
            "",
            "## Selected diagnostic runs",
            "",
            "| Vehicles | Training seed | Evaluation seed | Success | Detail archive |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            f"| {row['configured_vehicle_count']} | {row['training_seed']} | "
            f"{row['evaluation_seed']} | {100 * row['success_rate']:.2f}% | "
            f"`{row.get('task_record_archive', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Storage",
            "",
            (
                f"- Estimated raw detail size: "
                f"{storage['estimated_selected_raw_gb']:.2f} GiB"
            ),
            f"- Detail budget: {storage['diagnostic_detail_budget_gb']:.2f} GiB",
            "- Selected task records are stored as gzip CSV archives.",
            "- Exact aggregate metrics remain in every summary.json.",
            "- Every-step pricing diagnostics remain in every pricing.jsonl.",
            "",
        ]
    )
    return "\n".join(lines)


def _case_key(
    phase: str,
    vehicles: int,
    training_seed: int,
    evaluation_seed: int | None = None,
) -> str:
    suffix = (
        f":eval{evaluation_seed}" if evaluation_seed is not None else ""
    )
    return f"{phase}:v{vehicles}:train{training_seed}{suffix}"


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
