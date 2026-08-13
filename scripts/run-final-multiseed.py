from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tomllib

from vehicular_offloading.config import SimulationConfig


STRATEGIES = (
    "random",
    "greedy",
    "dqn",
    "stackelberg",
    "hybrid_stackelberg",
)
LEARNED_STRATEGIES = ("dqn", "hybrid_stackelberg")
REPORT_METRICS = (
    "success_rate",
    "avg_success_latency_s",
    "avg_latency_s",
    "avg_energy_j",
    "avg_cost_per_task",
    "avg_reward",
    "local_offload_ratio",
    "v2v_offload_ratio",
    "v2i_offload_ratio",
    "local_success_rate",
    "v2v_success_rate",
    "v2i_success_rate",
    "oracle_success_rate",
    "avoidable_failure_rate",
    "avg_decision_regret_s",
    "dqn_decision_ratio",
    "hybrid_deviation_ratio",
    "hybrid_beneficial_deviation_rate",
    "avg_cloud_queue_length",
    "max_predicted_cloud_capacity_ratio",
    "v2v_latency_advantage_ratio",
    "v2v_rescuable_task_ratio",
    "queue_induced_local_timeout_ratio",
    "wall_clock_s",
)
PAIRED_METRICS = (
    "success_rate",
    "avg_success_latency_s",
    "avg_energy_j",
    "avg_cost_per_task",
    "avg_reward",
)
T_CRITICAL_95 = {
    2: 12.706205,
    3: 4.302653,
    4: 3.182446,
    5: 2.776445,
    6: 2.570582,
    7: 2.446912,
    8: 2.364624,
    9: 2.306004,
    10: 2.262157,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and aggregate the final paired multi-seed experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    repository = _repository_root(config_path)
    with config_path.open("rb") as handle:
        pipeline = tomllib.load(handle)["pipeline"]
    _validate_pipeline(pipeline)
    parallelism = (
        int(args.parallelism)
        if args.parallelism is not None
        else int(pipeline["parallelism"])
    )
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")

    base_path = (
        args.base_config.resolve()
        if args.base_config is not None
        else (config_path.parent / pipeline["base_config"]).resolve()
    )
    base = SimulationConfig.from_toml(base_path)
    output = (repository / pipeline["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = _signature(base, pipeline)
    commit = _git_commit(repository) or "unknown"
    manifest_path = output / "experiment-manifest.json"
    _validate_or_create_manifest(
        manifest_path,
        commit,
        signature,
        pipeline,
        base_path,
        parallelism,
        args.dry_run,
    )

    total_training = (
        len(pipeline["training_seeds"])
        * len(pipeline["vehicle_counts"])
        * len(pipeline["training_strategies"])
    )
    total_evaluation = (
        len(pipeline["evaluation_seeds"])
        * len(pipeline["vehicle_counts"])
        * len(pipeline["evaluation_strategies"])
    )
    print(
        f"FINAL DESIGN: {total_training} training runs + "
        f"{total_evaluation} evaluations = "
        f"{total_training + total_evaluation} runs.",
        flush=True,
    )
    print(
        "Each training seed is paired with exactly one independent evaluation "
        "seed; no checkpoint selection is performed.",
        flush=True,
    )
    print(
        f"Task rows: training disabled, evaluation sample "
        f"{100 * float(pipeline['evaluation_task_sample_rate']):.3f}%.",
        flush=True,
    )
    _check_free_disk(output, float(pipeline["minimum_free_disk_gb"]))

    training_seeds = [int(value) for value in pipeline["training_seeds"]]
    evaluation_seeds = [int(value) for value in pipeline["evaluation_seeds"]]
    runner = repository / "scripts" / "run-training-evaluation.py"
    generated_paths: list[Path] = []
    replicate_count = len(training_seeds)
    concurrent_replicates, workers_per_replicate = _global_worker_allocation(
        replicate_count,
        parallelism,
    )
    print(
        "GLOBAL SCHEDULER: "
        f"{concurrent_replicates} replicate pipelines x "
        f"{workers_per_replicate} workers; "
        f"maximum active simulation workers="
        f"{concurrent_replicates * workers_per_replicate}.",
        flush=True,
    )
    try:
        replicate_commands = []
        for replicate, (training_seed, evaluation_seed) in enumerate(
            zip(training_seeds, evaluation_seeds, strict=True),
            start=1,
        ):
            generated = (
                repository
                / "configs"
                / f".final-multiseed-replicate-{replicate:02d}.toml"
            )
            generated_paths.append(generated)
            replicate_output = output / f"replicate-{replicate:02d}"
            generated.write_text(
                _render_replicate_config(
                    base_path,
                    replicate_output,
                    pipeline,
                    training_seed,
                    evaluation_seed,
                    workers_per_replicate,
                ),
                encoding="utf-8",
            )
            label = (
                f"replicate {replicate}/{len(training_seeds)} "
                f"train={training_seed} eval={evaluation_seed}"
            )
            command = [
                sys.executable,
                str(runner),
                "--config",
                str(generated),
                "--parallelism",
                str(workers_per_replicate),
            ]
            if args.dry_run:
                command.append("--dry-run")
            replicate_commands.append((replicate, label, command))
        with ThreadPoolExecutor(max_workers=concurrent_replicates) as executor:
            futures = {}
            for replicate, label, command in replicate_commands:
                print(f"BEGIN {label}", flush=True)
                future = executor.submit(
                    subprocess.run,
                    command,
                    cwd=repository,
                )
                futures[future] = (replicate, label)
            failures = []
            for future in as_completed(futures):
                replicate, label = futures[future]
                completed = future.result()
                if completed.returncode:
                    failures.append((replicate, label, completed.returncode))
                    print(
                        f"FAILED {label} code={completed.returncode}",
                        flush=True,
                    )
                    continue
                print(f"END {label}", flush=True)
                _check_free_disk(
                    output,
                    float(pipeline["minimum_free_disk_gb"]),
                )
            if failures:
                replicate, label, returncode = failures[0]
                raise RuntimeError(
                    f"{len(failures)} replicate pipeline(s) failed; first "
                    f"failure was {label} with code {returncode}. Rerun the "
                    "same command to resume completed cases."
                )
    finally:
        for path in generated_paths:
            path.unlink(missing_ok=True)

    if args.dry_run:
        print(
            "DRY RUN COMPLETE: all replicate configurations validated under "
            "the bounded global worker budget."
        )
        return 0

    rows = _collect_rows(output, training_seeds, evaluation_seeds)
    _validate_results(rows, pipeline)
    aggregates = _aggregate_results(rows)
    paired = _paired_comparisons(rows)
    _write_csv(output / "evaluation-results.csv", rows)
    _write_csv(output / "aggregate-results.csv", aggregates)
    _write_csv(output / "paired-comparisons.csv", paired)
    (output / "final-summary.md").write_text(
        _render_summary(pipeline, rows, aggregates, paired, commit, signature),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_at"] = _utc_now()
    manifest["completed_runs"] = total_training + total_evaluation
    manifest["result_files"] = [
        "evaluation-results.csv",
        "aggregate-results.csv",
        "paired-comparisons.csv",
        "final-summary.md",
    ]
    _write_json_atomic(manifest_path, manifest)
    print(f"COMPLETE {output}")
    print(f"SUMMARY {output / 'final-summary.md'}")
    return 0


def _global_worker_allocation(
    replicate_count: int,
    parallelism: int,
) -> tuple[int, int]:
    if replicate_count <= 0 or parallelism <= 0:
        raise ValueError("replicate_count and parallelism must be positive")
    concurrent_replicates = min(replicate_count, parallelism)
    workers_per_replicate = max(1, parallelism // concurrent_replicates)
    return concurrent_replicates, workers_per_replicate


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate.resolve()
    raise RuntimeError(f"could not locate repository root from {config_path}")


def _validate_pipeline(pipeline: dict) -> None:
    required = {
        "base_config",
        "output_dir",
        "training_steps",
        "evaluation_steps",
        "vehicle_counts",
        "training_seeds",
        "evaluation_seeds",
        "training_strategies",
        "evaluation_strategies",
        "training_task_sample_rate",
        "evaluation_task_sample_rate",
        "parallelism",
        "minimum_free_disk_gb",
        "storage_upper_bound_gb",
    }
    missing = required - set(pipeline)
    if missing:
        raise ValueError(f"missing pipeline settings: {sorted(missing)}")
    training_seeds = [int(value) for value in pipeline["training_seeds"]]
    evaluation_seeds = [int(value) for value in pipeline["evaluation_seeds"]]
    if len(training_seeds) < 2 or len(training_seeds) != len(evaluation_seeds):
        raise ValueError(
            "training_seeds and evaluation_seeds must contain the same "
            "number of at least two paired seeds"
        )
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be unique")
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("evaluation_seeds must be unique")
    training_strategies = tuple(pipeline["training_strategies"])
    evaluation_strategies = tuple(pipeline["evaluation_strategies"])
    allow_subset = bool(pipeline.get("allow_strategy_subset", False))
    if not allow_subset:
        if training_strategies != LEARNED_STRATEGIES:
            raise ValueError(
                "training_strategies must be dqn and hybrid_stackelberg"
            )
        if evaluation_strategies != STRATEGIES:
            raise ValueError(
                "evaluation_strategies must contain the five fixed strategies"
            )
    else:
        if not training_strategies or not set(training_strategies) <= set(LEARNED_STRATEGIES):
            raise ValueError("subset training_strategies must be learned strategies")
        if not evaluation_strategies or not set(evaluation_strategies) <= set(STRATEGIES):
            raise ValueError("subset evaluation_strategies contain an unknown strategy")
        hybrid_source = str(
            pipeline.get("hybrid_checkpoint_strategy", "hybrid_stackelberg")
        )
        required = {
            hybrid_source if strategy == "hybrid_stackelberg" else strategy
            for strategy in evaluation_strategies
            if strategy in LEARNED_STRATEGIES
        }
        if not required <= set(training_strategies):
            raise ValueError(
                "each learned evaluation strategy requires a trained checkpoint"
            )
    if min(int(value) for value in pipeline["vehicle_counts"]) <= 0:
        raise ValueError("vehicle_counts must be positive")
    if min(int(pipeline["training_steps"]), int(pipeline["evaluation_steps"])) <= 0:
        raise ValueError("training_steps and evaluation_steps must be positive")
    for key in ("training_task_sample_rate", "evaluation_task_sample_rate"):
        rate = float(pipeline[key])
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    if float(pipeline["minimum_free_disk_gb"]) < 0.0:
        raise ValueError("minimum_free_disk_gb must be non-negative")
    if float(pipeline["storage_upper_bound_gb"]) <= 0.0:
        raise ValueError("storage_upper_bound_gb must be positive")


def _render_replicate_config(
    base_path: Path,
    output: Path,
    pipeline: dict,
    training_seed: int,
    evaluation_seed: int,
    parallelism: int,
) -> str:
    values = [
        "[pipeline]",
        f"base_config = {json.dumps(base_path.as_posix())}",
        f"output_dir = {json.dumps(output.as_posix())}",
        f"training_steps = {int(pipeline['training_steps'])}",
        f"evaluation_steps = {int(pipeline['evaluation_steps'])}",
        f"vehicle_counts = {json.dumps(pipeline['vehicle_counts'])}",
        f"training_seed = {training_seed}",
        f"evaluation_seed = {evaluation_seed}",
        f"training_strategies = {json.dumps(pipeline['training_strategies'])}",
        f"evaluation_strategies = {json.dumps(pipeline['evaluation_strategies'])}",
        *(
            [
                "hybrid_checkpoint_strategy = "
                f"{json.dumps(pipeline['hybrid_checkpoint_strategy'])}"
            ]
            if "hybrid_checkpoint_strategy" in pipeline
            else []
        ),
        (
            "training_task_sample_rate = "
            f"{float(pipeline['training_task_sample_rate'])}"
        ),
        (
            "evaluation_task_sample_rate = "
            f"{float(pipeline['evaluation_task_sample_rate'])}"
        ),
        f"parallelism = {parallelism}",
        "",
    ]
    return "\n".join(values)


def _collect_rows(
    output: Path,
    training_seeds: list[int],
    evaluation_seeds: list[int],
) -> list[dict]:
    rows: list[dict] = []
    for replicate, (training_seed, evaluation_seed) in enumerate(
        zip(training_seeds, evaluation_seeds, strict=True),
        start=1,
    ):
        replicate_root = output / f"replicate-{replicate:02d}"
        candidates = sorted(
            replicate_root.glob("run-*/evaluation-results.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"evaluation-results.csv is missing for replicate {replicate}"
            )
        with candidates[0].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "replicate": replicate,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        **row,
                    }
                )
    return rows


def _validate_results(rows: list[dict], pipeline: dict) -> None:
    evaluation_strategies = tuple(pipeline.get("evaluation_strategies", STRATEGIES))
    expected = (
        len(pipeline["training_seeds"])
        * len(pipeline["vehicle_counts"])
        * len(evaluation_strategies)
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} evaluation rows, found {len(rows)}")
    by_case: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        replicate = int(row["replicate"])
        vehicles = int(row["configured_vehicle_count"])
        by_case.setdefault((replicate, vehicles), []).append(row)
        success = float(row["success_rate"])
        if not 0.0 <= success <= 1.0:
            raise RuntimeError(f"success_rate outside [0, 1]: {row}")
        ratio_sum = sum(
            float(row[key])
            for key in (
                "local_offload_ratio",
                "v2v_offload_ratio",
                "v2i_offload_ratio",
            )
        )
        if not math.isclose(ratio_sum, 1.0, abs_tol=1e-9):
            raise RuntimeError(f"offload ratios do not sum to one: {row}")
        if int(row["completed_steps"]) != int(pipeline["evaluation_steps"]):
            raise RuntimeError(f"incomplete evaluation: {row['run_dir']}")
    for case, case_rows in by_case.items():
        observed = {row["strategy"] for row in case_rows}
        if observed != set(evaluation_strategies):
            raise RuntimeError(f"strategy set mismatch for {case}: {sorted(observed)}")
        task_counts = {int(row["total_tasks"]) for row in case_rows}
        if len(task_counts) != 1:
            raise RuntimeError(
                f"task streams differ across strategies for {case}: "
                f"{sorted(task_counts)}"
            )


def _aggregate_results(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["configured_vehicle_count"]), row["strategy"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (vehicles, strategy), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], STRATEGIES.index(item[0][1]))
    ):
        aggregate: dict[str, int | float | str] = {
            "vehicle_count": vehicles,
            "strategy": strategy,
            "replicates": len(group),
        }
        for metric in REPORT_METRICS:
            values = [float(row[metric]) for row in group]
            mean_value, sample_std, ci95 = _mean_std_ci(values)
            aggregate[f"{metric}_mean"] = mean_value
            aggregate[f"{metric}_sample_std"] = sample_std
            aggregate[f"{metric}_ci95"] = ci95
        aggregates.append(aggregate)
    return aggregates


def _paired_comparisons(rows: list[dict]) -> list[dict]:
    observed_strategies = {row["strategy"] for row in rows}
    if "hybrid_stackelberg" not in observed_strategies:
        return []
    indexed = {
        (
            int(row["replicate"]),
            int(row["configured_vehicle_count"]),
            row["strategy"],
        ): row
        for row in rows
    }
    vehicles = sorted({int(row["configured_vehicle_count"]) for row in rows})
    replicates = sorted({int(row["replicate"]) for row in rows})
    comparisons: list[dict] = []
    for count in vehicles:
        for comparator in STRATEGIES:
            if comparator == "hybrid_stackelberg" or comparator not in observed_strategies:
                continue
            result: dict[str, int | float | str] = {
                "vehicle_count": count,
                "comparison": f"hybrid_stackelberg-minus-{comparator}",
                "replicates": len(replicates),
            }
            for metric in PAIRED_METRICS:
                differences = [
                    float(indexed[(replicate, count, "hybrid_stackelberg")][metric])
                    - float(indexed[(replicate, count, comparator)][metric])
                    for replicate in replicates
                ]
                mean_value, sample_std, ci95 = _mean_std_ci(differences)
                result[f"{metric}_mean_delta"] = mean_value
                result[f"{metric}_sample_std"] = sample_std
                result[f"{metric}_ci95"] = ci95
            comparisons.append(result)
    return comparisons


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty sample")
    mean_value = statistics.fmean(values)
    if len(values) == 1:
        return mean_value, 0.0, 0.0
    sample_std = statistics.stdev(values)
    critical = T_CRITICAL_95.get(len(values), 1.959964)
    ci95 = critical * sample_std / math.sqrt(len(values))
    return mean_value, sample_std, ci95


def _render_summary(
    pipeline: dict,
    rows: list[dict],
    aggregates: list[dict],
    paired: list[dict],
    commit: str,
    signature: str,
) -> str:
    lines = [
        "# Final paired multi-seed experiment",
        "",
        f"- Git commit: `{commit}`",
        f"- Experiment signature: `{signature}`",
        f"- Training seeds: {', '.join(map(str, pipeline['training_seeds']))}",
        f"- Evaluation seeds: {', '.join(map(str, pipeline['evaluation_seeds']))}",
        (
            "- Design: one independently trained checkpoint per strategy, vehicle "
            "scale, and paired replicate; no best-seed selection."
        ),
        (
            "- Intervals: two-sided 95% Student t intervals over the "
            f"paired replicates (n={len(pipeline['training_seeds'])})."
        ),
        f"- Evaluation runs: {len(rows)}",
        "",
        "## Aggregate results",
        "",
        (
            "| Vehicles | Strategy | Success (95% CI) | Success latency (s) | "
            "Energy (J) | Cost/task | Reward |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['vehicle_count']} | {row['strategy']} | "
            f"{100 * float(row['success_rate_mean']):.2f}% ± "
            f"{100 * float(row['success_rate_ci95']):.2f} pp | "
            f"{float(row['avg_success_latency_s_mean']):.4f} | "
            f"{float(row['avg_energy_j_mean']):.2f} | "
            f"{float(row['avg_cost_per_task_mean']):.4f} | "
            f"{float(row['avg_reward_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Hybrid success differences",
            "",
            "| Vehicles | Comparator | Hybrid minus comparator (95% CI) |",
            "|---:|---|---:|",
        ]
    )
    for row in paired:
        comparator = str(row["comparison"]).removeprefix(
            "hybrid_stackelberg-minus-"
        )
        lines.append(
            f"| {row['vehicle_count']} | {comparator} | "
            f"{100 * float(row['success_rate_mean_delta']):+.2f} ± "
            f"{100 * float(row['success_rate_ci95']):.2f} pp |"
        )
    lines.extend(
        [
            "",
            "A positive success delta favors Hybrid. For latency, energy, and "
            "cost deltas in `paired-comparisons.csv`, negative values favor "
            "Hybrid; for reward, positive values favor Hybrid.",
            "",
            "Exact per-run metrics and run directories are preserved in "
            "`evaluation-results.csv`; sampled task records are diagnostic only "
            "and do not affect aggregates.",
            "",
        ]
    )
    return "\n".join(lines)


def _signature(base: SimulationConfig, pipeline: dict) -> str:
    encoded = json.dumps(
        {"base": base.to_dict(), "pipeline": pipeline},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


def _validate_or_create_manifest(
    path: Path,
    commit: str,
    signature: str,
    pipeline: dict,
    base_path: Path,
    parallelism: int,
    dry_run: bool,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("git_commit") != commit
            or existing.get("experiment_signature") != signature
        ):
            raise RuntimeError(
                "existing final experiment belongs to a different commit or "
                "configuration; use -Reset only after reviewing the exact path"
            )
        return
    if dry_run:
        return
    _write_json_atomic(
        path,
        {
            "created_at": _utc_now(),
            "git_commit": commit,
            "experiment_signature": signature,
            "base_config": str(base_path),
            "parallelism": parallelism,
            "pipeline": pipeline,
        },
    )


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_free_disk(path: Path, minimum_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB is free; "
            f"minimum_free_disk_gb is {minimum_gb:.2f}"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value: dict) -> None:
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
