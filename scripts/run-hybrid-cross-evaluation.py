from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import (
    TraceCachingMobilityProvider,
    create_mobility,
)
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


VEHICLE_COUNTS = (2_000, 4_000)
RESULT_FIELDS = [
    "training_seed",
    "evaluation_seed",
    "vehicle_count",
    "source",
    "success_rate",
    "avg_latency_s",
    "avg_success_latency_s",
    "avg_energy_j",
    "avg_cost_per_task",
    "avg_reward",
    "local_offload_ratio",
    "v2v_offload_ratio",
    "v2i_offload_ratio",
    "avg_cloud_queue_length",
    "avg_source_workload_s",
    "hybrid_deviation_ratio",
    "hybrid_beneficial_deviation_rate",
    "all_actions_late_rate",
    "wall_clock_s",
    "run_dir",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate Hybrid checkpoints and evaluation seeds"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.parallelism <= 0:
        raise ValueError("parallelism must be positive")

    base = SimulationConfig.from_toml(args.config.resolve())
    sessions = _load_sessions(args.source)
    if len(sessions) != 3:
        raise RuntimeError(
            f"expected three source sessions, found {len(sessions)}"
        )
    training_seeds = sorted(sessions)
    evaluation_seeds = sorted(
        {session["evaluation_seed"] for session in sessions.values()}
    )
    if len(evaluation_seeds) != 3:
        raise RuntimeError(
            f"expected three evaluation seeds, found {evaluation_seeds}"
        )

    output = args.output_dir.resolve()
    cases = []
    for training_seed in training_seeds:
        source = sessions[training_seed]
        for evaluation_seed in evaluation_seeds:
            if evaluation_seed == source["evaluation_seed"]:
                continue
            for vehicles in VEHICLE_COUNTS:
                config = _evaluation_config(
                    base,
                    vehicles,
                    evaluation_seed,
                    source["checkpoints"][vehicles],
                    output,
                    training_seed,
                )
                cases.append((training_seed, evaluation_seed, vehicles, config))

    if args.dry_run:
        for training_seed, evaluation_seed, vehicles, config in cases:
            config.validate()
            print(
                f"DRY train={training_seed} eval={evaluation_seed} "
                f"vehicles={vehicles}"
            )
        print(f"DRY RUN OK: {len(cases)} off-diagonal evaluations")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    pending = []
    off_diagonal_rows = []
    for index, case in enumerate(cases, start=1):
        recovered = _recover(case)
        if recovered is None:
            pending.append((index, *case))
        else:
            off_diagonal_rows.append(recovered)
            print(
                f"SKIP {index}/{len(cases)} train={case[0]} "
                f"eval={case[1]} vehicles={case[2]}",
                flush=True,
            )

    _prepare_shared_scenarios(pending)
    _prepare_shared_mobility(pending)
    failures = []
    workers = min(args.parallelism, len(pending))
    if workers:
        scheduled = sorted(
            pending, key=lambda item: item[3], reverse=True
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, training_seed, evaluation_seed, vehicles, config in scheduled:
                print(
                    f"START {index}/{len(cases)} train={training_seed} "
                    f"eval={evaluation_seed} vehicles={vehicles}",
                    flush=True,
                )
                future = executor.submit(
                    _execute,
                    config,
                    training_seed,
                    evaluation_seed,
                    vehicles,
                )
                futures[future] = (
                    index,
                    training_seed,
                    evaluation_seed,
                    vehicles,
                )
            for future in as_completed(futures):
                index, training_seed, evaluation_seed, vehicles = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    failures.append(
                        (index, training_seed, evaluation_seed, vehicles, error)
                    )
                    print(
                        f"FAILED {index}/{len(cases)} train={training_seed} "
                        f"eval={evaluation_seed} vehicles={vehicles}: {error}",
                        flush=True,
                    )
                    continue
                off_diagonal_rows.append(row)
                _write_csv(
                    output / "off-diagonal-results.csv",
                    sorted(off_diagonal_rows, key=_row_key),
                )
                print(
                    f"DONE {index}/{len(cases)} train={training_seed} "
                    f"eval={evaluation_seed} vehicles={vehicles} "
                    f"success={100 * row['success_rate']:.2f}% "
                    f"wall={row['wall_clock_s']:.1f}s",
                    flush=True,
                )
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"{len(failures)} cross-evaluation run(s) failed; first failure "
            f"was train={first[1]} eval={first[2]} vehicles={first[3]}"
        ) from first[4]

    diagonal_rows = _diagonal_rows(sessions)
    matrix_rows = sorted(diagonal_rows + off_diagonal_rows, key=_row_key)
    expected = len(training_seeds) * len(evaluation_seeds) * len(VEHICLE_COUNTS)
    if len(matrix_rows) != expected:
        raise RuntimeError(
            f"expected {expected} matrix rows, found {len(matrix_rows)}"
        )
    _write_csv(output / "cross-evaluation-matrix.csv", matrix_rows)
    analysis = _analyze(matrix_rows, training_seeds, evaluation_seeds)
    (output / "cross-evaluation-summary.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "cross-evaluation-summary.md").write_text(
        _markdown(analysis, training_seeds, evaluation_seeds),
        encoding="utf-8",
    )
    print(f"COMPLETE {output}")
    print(f"SUMMARY {output / 'cross-evaluation-summary.md'}")
    return 0


def _load_sessions(roots: list[Path]) -> dict[int, dict]:
    sessions = {}
    for root in roots:
        for training_path in root.resolve().rglob("training-results.csv"):
            session = training_path.parent
            evaluation_path = session / "evaluation-results.csv"
            diagnostic_path = session / "diagnostic-summary.json"
            if not evaluation_path.exists() or not diagnostic_path.exists():
                continue
            training_rows = _read_csv(training_path)
            training_seeds = {int(row["seed"]) for row in training_rows}
            if len(training_seeds) != 1:
                continue
            training_seed = training_seeds.pop()
            evaluation_rows = _read_csv(evaluation_path)
            evaluation_seeds = {int(row["seed"]) for row in evaluation_rows}
            if len(evaluation_seeds) != 1:
                continue
            checkpoints = {}
            for row in training_rows:
                if (
                    row["strategy"] == "hybrid_stackelberg"
                    and int(row["configured_vehicle_count"]) in VEHICLE_COUNTS
                ):
                    checkpoint = Path(row["run_dir"]) / "dqn-policy.pt"
                    if not checkpoint.exists():
                        raise RuntimeError(f"missing checkpoint: {checkpoint}")
                    checkpoints[int(row["configured_vehicle_count"])] = checkpoint
            if set(checkpoints) != set(VEHICLE_COUNTS):
                continue
            if training_seed in sessions:
                raise RuntimeError(
                    f"duplicate source session for training seed {training_seed}"
                )
            sessions[training_seed] = {
                "path": session,
                "evaluation_seed": evaluation_seeds.pop(),
                "checkpoints": checkpoints,
                "evaluation_rows": evaluation_rows,
            }
    return sessions


def _evaluation_config(
    base: SimulationConfig,
    vehicles: int,
    evaluation_seed: int,
    checkpoint: Path,
    output: Path,
    training_seed: int,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = "hybrid_stackelberg"
    config.vehicle_count = vehicles
    config.steps = 2_000
    config.seed = evaluation_seed
    config.output_dir = str(
        output
        / "runs"
        / f"train-{training_seed}-eval-{evaluation_seed}-vehicles-{vehicles}"
    )
    config.record_decision_diagnostics = True
    config.record_task_records = True
    config.dqn.mode = "evaluate"
    config.dqn.checkpoint_path = str(checkpoint.resolve())
    config.validate()
    return config


def _prepare_shared_scenarios(pending: list[tuple]) -> None:
    prepared = {}
    for _index, _train, _evaluation, _vehicles, config in pending:
        if config.mobility != "sumo" or not config.scenario_net:
            continue
        key = (
            config.scenario_net,
            config.route_output_dir,
            config.vehicle_count,
            config.route_departure_end_s,
            config.seed,
        )
        scenario = prepared.get(key)
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
            prepared[key] = scenario
        config.scenario_config = scenario
        config.scenario_net = None


def _prepare_shared_mobility(pending: list[tuple]) -> None:
    prepared = set()
    for _index, _train, evaluation_seed, vehicles, config in pending:
        if config.mobility != "sumo":
            continue
        mobility = create_mobility(config)
        if not isinstance(mobility, TraceCachingMobilityProvider):
            continue
        cache_path = mobility.cache_path.resolve()
        if cache_path in prepared or mobility.cache_is_valid():
            prepared.add(cache_path)
            continue
        print(
            f"PREPARE MOBILITY vehicles={vehicles} seed={evaluation_seed}",
            flush=True,
        )
        mobility.start()
        try:
            for step in range(config.steps):
                mobility.step(step)
        finally:
            mobility.close()
        prepared.add(cache_path)


def _execute(
    config: SimulationConfig,
    training_seed: int,
    evaluation_seed: int,
    vehicles: int,
) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    wall_s = perf_counter() - wall_started
    _ = process_time() - cpu_started
    row = {
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "vehicle_count": vehicles,
        "source": "cross_evaluation",
        **asdict(summary),
        "wall_clock_s": wall_s,
        "run_dir": str(run_dir),
    }
    return _select_result_fields(row)


def _recover(case: tuple) -> dict | None:
    training_seed, evaluation_seed, vehicles, config = case
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
        config_path = run_dir / "config.json"
        if not all(path.exists() for path in (summary_path, timing_path, config_path)):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            int(summary.get("seed", -1)) != evaluation_seed
            or int(summary.get("configured_vehicle_count", -1)) != vehicles
            or int(summary.get("completed_steps", -1)) != 2_000
            or Path(saved_config["dqn"]["checkpoint_path"]).resolve()
            != Path(config.dqn.checkpoint_path).resolve()
        ):
            continue
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        return _select_result_fields(
            {
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "vehicle_count": vehicles,
                "source": "cross_evaluation",
                **summary,
                "wall_clock_s": float(timing["wall_clock_s"]),
                "run_dir": str(run_dir),
            }
        )
    return None


def _diagonal_rows(sessions: dict[int, dict]) -> list[dict]:
    rows = []
    for training_seed, session in sessions.items():
        for row in session["evaluation_rows"]:
            vehicles = int(row["configured_vehicle_count"])
            if row["strategy"] != "hybrid_stackelberg" or vehicles not in VEHICLE_COUNTS:
                continue
            rows.append(
                _select_result_fields(
                    {
                        **row,
                        "training_seed": training_seed,
                        "evaluation_seed": int(row["seed"]),
                        "vehicle_count": vehicles,
                        "source": "existing_diagonal",
                    }
                )
            )
    return rows


def _select_result_fields(row: dict) -> dict:
    selected = {}
    integer_fields = {
        "training_seed",
        "evaluation_seed",
        "vehicle_count",
    }
    text_fields = {"source", "run_dir"}
    for field in RESULT_FIELDS:
        value = row[field]
        if field in integer_fields:
            selected[field] = int(value)
        elif field in text_fields:
            selected[field] = str(value)
        else:
            selected[field] = float(value)
    return selected


def _analyze(
    rows: list[dict],
    training_seeds: list[int],
    evaluation_seeds: list[int],
) -> dict:
    result = {"vehicle_counts": {}}
    for vehicles in VEHICLE_COUNTS:
        selected = [row for row in rows if row["vehicle_count"] == vehicles]
        values = {
            (row["training_seed"], row["evaluation_seed"]): row["success_rate"]
            for row in selected
        }
        grand = mean(values.values())
        train_means = {
            seed: mean(values[(seed, evaluation)] for evaluation in evaluation_seeds)
            for seed in training_seeds
        }
        evaluation_means = {
            seed: mean(values[(training, seed)] for training in training_seeds)
            for seed in evaluation_seeds
        }
        ss_training = len(evaluation_seeds) * sum(
            (value - grand) ** 2 for value in train_means.values()
        )
        ss_evaluation = len(training_seeds) * sum(
            (value - grand) ** 2 for value in evaluation_means.values()
        )
        ss_interaction = sum(
            (
                values[(training, evaluation)]
                - train_means[training]
                - evaluation_means[evaluation]
                + grand
            )
            ** 2
            for training in training_seeds
            for evaluation in evaluation_seeds
        )
        total = ss_training + ss_evaluation + ss_interaction
        result["vehicle_counts"][str(vehicles)] = {
            "matrix": {
                str(training): {
                    str(evaluation): values[(training, evaluation)]
                    for evaluation in evaluation_seeds
                }
                for training in training_seeds
            },
            "grand_mean": grand,
            "training_seed_means": train_means,
            "evaluation_seed_means": evaluation_means,
            "sum_of_squares": {
                "training_seed": ss_training,
                "evaluation_seed": ss_evaluation,
                "interaction": ss_interaction,
            },
            "sum_of_squares_share": {
                "training_seed": ss_training / total if total else 0.0,
                "evaluation_seed": ss_evaluation / total if total else 0.0,
                "interaction": ss_interaction / total if total else 0.0,
            },
        }
    return result


def _markdown(
    analysis: dict,
    training_seeds: list[int],
    evaluation_seeds: list[int],
) -> str:
    lines = [
        "# Hybrid training/evaluation seed cross-evaluation",
        "",
        "Rows are independently trained Hybrid checkpoints. Columns are independent "
        "evaluation mobility and task streams.",
    ]
    for vehicles in VEHICLE_COUNTS:
        item = analysis["vehicle_counts"][str(vehicles)]
        lines.extend(
            [
                "",
                f"## {vehicles} vehicles",
                "",
                "| Training seed | "
                + " | ".join(f"Eval {seed}" for seed in evaluation_seeds)
                + " | Row mean |",
                "|---:|" + "---:|" * (len(evaluation_seeds) + 1),
            ]
        )
        for training_seed in training_seeds:
            matrix_row = item["matrix"][str(training_seed)]
            values = [
                100 * matrix_row[str(seed)] for seed in evaluation_seeds
            ]
            lines.append(
                f"| {training_seed} | "
                + " | ".join(f"{value:.2f}%" for value in values)
                + f" | {mean(values):.2f}% |"
            )
        column_means = item["evaluation_seed_means"]
        lines.append(
            "| Column mean | "
            + " | ".join(
                f"{100 * column_means[seed]:.2f}%" for seed in evaluation_seeds
            )
            + f" | {100 * item['grand_mean']:.2f}% |"
        )
        shares = item["sum_of_squares_share"]
        lines.extend(
            [
                "",
                "Variation share (two-way sum of squares):",
                "",
                f"- Training seed: {100 * shares['training_seed']:.2f}%",
                f"- Evaluation seed: {100 * shares['evaluation_seed']:.2f}%",
                f"- Training × evaluation interaction: "
                f"{100 * shares['interaction']:.2f}%",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _row_key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["vehicle_count"]),
        int(row["training_seed"]),
        int(row["evaluation_seed"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
