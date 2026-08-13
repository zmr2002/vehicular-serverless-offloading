from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tomllib


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a paired final-model Knative validation config"
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        default=Path("configs/final-three-seed.toml"),
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/.knative-final-three-seed.generated.toml"),
    )
    parser.add_argument(
        "--validation-output-dir",
        type=Path,
        default=Path("results/verified/serverless-final-model-three-seed"),
    )
    parser.add_argument("--analytical-parallelism", type=int, default=6)
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        help=(
            "One-based final-experiment replicates to validate. "
            "The default validates every completed replicate."
        ),
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    final_config = _resolve(repository, args.final_config)
    with final_config.open("rb") as handle:
        pipeline = tomllib.load(handle)["pipeline"]
    output_root = _resolve(repository, Path(pipeline["output_dir"]))
    manifest = _completed_manifest(output_root)
    evaluation_seeds = [int(value) for value in pipeline["evaluation_seeds"]]
    vehicle_counts = [int(value) for value in pipeline["vehicle_counts"]]
    evaluation_steps = int(pipeline["evaluation_steps"])
    if len(evaluation_seeds) != len(pipeline["training_seeds"]):
        raise ValueError("training and evaluation seeds must remain paired")
    selected_replicates = _select_replicates(
        evaluation_seeds,
        args.replicates,
    )
    checkpoint_strategy = str(
        pipeline.get("hybrid_checkpoint_strategy", "hybrid_stackelberg")
    )
    if checkpoint_strategy not in {"dqn", "hybrid_stackelberg"}:
        raise ValueError(
            "hybrid_checkpoint_strategy must be dqn or hybrid_stackelberg"
        )

    if args.base_config is not None:
        base_config = _resolve(repository, args.base_config)
    else:
        base_config = _resolve(final_config.parent, Path(pipeline["base_config"]))
    if not base_config.is_file():
        raise FileNotFoundError(f"final base config not found: {base_config}")

    checkpoints: dict[tuple[int, int], Path] = {}
    for replicate, seed in selected_replicates:
        replicate_root = output_root / f"replicate-{replicate:02d}"
        for vehicles in vehicle_counts:
            checkpoints[(vehicles, seed)] = _discover_checkpoint(
                replicate_root,
                vehicles,
                checkpoint_strategy,
            )

    task_estimates, request_budgets = _final_evaluation_statistics(
        output_root,
        selected_replicates,
        vehicle_counts,
    )

    output = _resolve(repository, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render(
            repository,
            base_config,
            vehicle_counts,
            [seed for _, seed in selected_replicates],
            checkpoints,
            evaluation_steps=evaluation_steps,
            validation_output_dir=args.validation_output_dir,
            analytical_parallelism=args.analytical_parallelism,
            source_experiment=output_root,
            source_commit=str(manifest.get("git_commit", "unknown")),
            checkpoint_strategy=checkpoint_strategy,
            replicate_numbers=[replicate for replicate, _ in selected_replicates],
            estimated_task_counts=task_estimates,
            request_budgets=request_budgets,
        ),
        encoding="utf-8",
    )
    print(f"GENERATED {output}")
    print(
        f"SOURCE {output_root} commit={manifest.get('git_commit', 'unknown')} "
        f"replicates={','.join(str(value) for value, _ in selected_replicates)} "
        f"checkpoint_strategy={checkpoint_strategy}"
    )
    return 0


def _completed_manifest(output_root: Path) -> dict:
    path = output_root / "experiment-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"final experiment manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("completed_at"):
        raise RuntimeError(f"final experiment is not complete: {path}")
    return manifest


def _select_replicates(
    evaluation_seeds: list[int],
    requested: list[int] | None,
) -> list[tuple[int, int]]:
    selected = requested or list(range(1, len(evaluation_seeds) + 1))
    if len(set(selected)) != len(selected):
        raise ValueError("replicate numbers must be unique")
    if any(value < 1 or value > len(evaluation_seeds) for value in selected):
        raise ValueError(
            f"replicate numbers must be within 1..{len(evaluation_seeds)}"
        )
    return [(replicate, evaluation_seeds[replicate - 1]) for replicate in selected]


def _discover_checkpoint(
    replicate_root: Path,
    vehicles: int,
    checkpoint_strategy: str = "hybrid_stackelberg",
) -> Path:
    sessions = sorted(
        replicate_root.glob("run-*/pipeline-state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for state_path in sessions:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved = state.get("runs", {}).get(
            f"train:{checkpoint_strategy}:{vehicles}"
        )
        if not saved:
            continue
        checkpoint = Path(saved["checkpoint"]).resolve()
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError(
        f"completed {checkpoint_strategy} checkpoint not found under "
        f"{replicate_root} for {vehicles} vehicles"
    )


def _final_evaluation_statistics(
    output_root: Path,
    selected_replicates: list[tuple[int, int]],
    vehicle_counts: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    path = output_root / "evaluation-results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"final evaluation results not found: {path}")
    selected = {replicate for replicate, _ in selected_replicates}
    tasks: dict[int, list[int]] = {vehicles: [] for vehicles in vehicle_counts}
    requests: dict[int, list[int]] = {vehicles: [] for vehicles in vehicle_counts}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") not in {"evaluate", "evaluation"}:
                continue
            if row.get("strategy") != "hybrid_stackelberg":
                continue
            replicate = int(row["replicate"])
            vehicles = int(row["configured_vehicle_count"])
            if replicate not in selected or vehicles not in tasks:
                continue
            total_tasks = int(row["total_tasks"])
            tasks[vehicles].append(total_tasks)
            requests[vehicles].append(
                math.ceil(total_tasks * float(row["v2i_offload_ratio"]))
            )
    expected = len(selected_replicates)
    missing = [
        vehicles
        for vehicles in vehicle_counts
        if len(tasks[vehicles]) != expected
    ]
    if missing:
        raise RuntimeError(
            "final Hybrid evaluation rows are incomplete for vehicle counts: "
            + ", ".join(str(value) for value in missing)
        )
    task_estimates = {vehicles: max(values) for vehicles, values in tasks.items()}
    # The analytical run is deterministic for the frozen input.  A 50% margin
    # permits closed-loop routing to react to measured HTTP delay while still
    # stopping an accidental all-cloud request storm.
    request_budgets = {
        vehicles: math.ceil(max(requests[vehicles]) * 1.5 + 10_000)
        for vehicles in vehicle_counts
    }
    return task_estimates, request_budgets


def _render(
    repository: Path,
    base_config: Path,
    vehicle_counts: list[int],
    seeds: list[int],
    checkpoints: dict[tuple[int, int], Path],
    *,
    evaluation_steps: int = 2000,
    validation_output_dir: Path = Path(
        "results/verified/serverless-final-model-three-seed"
    ),
    analytical_parallelism: int = 6,
    source_experiment: Path | None = None,
    source_commit: str | None = None,
    checkpoint_strategy: str | None = None,
    replicate_numbers: list[int] | None = None,
    estimated_task_counts: dict[int, int] | None = None,
    request_budgets: dict[int, int] | None = None,
) -> str:
    def relative(path: Path) -> str:
        return path.resolve().relative_to(repository.resolve()).as_posix()

    try:
        base_config_value = base_config.resolve().relative_to(
            (repository / "configs").resolve()
        ).as_posix()
    except ValueError:
        # Generated winning profiles live with the experiment artifacts rather
        # than under configs/.  The validation loader accepts an absolute path.
        base_config_value = base_config.resolve().as_posix()
    validation_output = (
        validation_output_dir.resolve().as_posix()
        if validation_output_dir.is_absolute()
        else validation_output_dir.as_posix()
    )
    estimated_task_counts = estimated_task_counts or {
        1000: 177402,
        2000: 640195,
        4000: 1661380,
    }
    request_budgets = request_budgets or {
        1000: 200000,
        2000: 700000,
        4000: 1800000,
    }

    lines = [
        "[validation]",
        f"base_config = {json.dumps(base_config_value)}",
        f"output_dir = {json.dumps(validation_output)}",
        f"steps = [{evaluation_steps}]",
        f"vehicle_counts = {json.dumps(vehicle_counts)}",
        f"seeds = {json.dumps(seeds)}",
        'modes = ["analytical", "knative_replay", "knative_closed_loop"]',
        'strategy = "hybrid_stackelberg"',
        "client_concurrency = 100",
        f"analytical_parallelism = {analytical_parallelism}",
        "task_record_sample_rate = 0.001",
        "minimum_free_disk_gb = 12.0",
        f"estimated_task_count_steps = {evaluation_steps}",
        "scale_to_zero_timeout_s = 360",
        "pod_poll_interval_s = 1.0",
    ]
    if source_experiment is not None:
        lines.append(
            f"source_experiment = {json.dumps(relative(source_experiment))}"
        )
    if source_commit is not None:
        lines.append(f"source_commit = {json.dumps(source_commit)}")
    if checkpoint_strategy is not None:
        lines.append(
            f"checkpoint_strategy = {json.dumps(checkpoint_strategy)}"
        )
    if replicate_numbers is not None:
        lines.append(f"replicates = {json.dumps(replicate_numbers)}")
    lines.extend(["", "[validation.checkpoints]"])
    for (vehicles, seed), checkpoint in sorted(checkpoints.items()):
        lines.append(
            f'{json.dumps(f"{vehicles}:{seed}")} = '
            f"{json.dumps(relative(checkpoint))}"
        )
    lines.extend(["", "[validation.request_budgets]"])
    for vehicles in vehicle_counts:
        lines.append(f'{json.dumps(str(vehicles))} = {request_budgets[vehicles]}')
    lines.extend(["", "[validation.estimated_task_counts]"])
    for vehicles in vehicle_counts:
        lines.append(
            f'{json.dumps(str(vehicles))} = {estimated_task_counts[vehicles]}'
        )
    lines.append("")
    return "\n".join(lines)


def _resolve(base: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
