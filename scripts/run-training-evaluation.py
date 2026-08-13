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
import subprocess
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import (
    TraceCachingMobilityProvider,
    create_mobility,
)
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


LEARNED_STRATEGIES = {"dqn", "hybrid_stackelberg"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train policies, freeze them, and run diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    pipeline_path = args.config.resolve()
    with pipeline_path.open("rb") as handle:
        pipeline = tomllib.load(handle)["pipeline"]
    if args.training_seed is not None:
        pipeline["training_seed"] = args.training_seed
    if args.evaluation_seed is not None:
        pipeline["evaluation_seed"] = args.evaluation_seed
    if args.output_dir is not None:
        pipeline["output_dir"] = args.output_dir
    base_path = (pipeline_path.parent / pipeline["base_config"]).resolve()
    base = SimulationConfig.from_toml(base_path)
    _validate_pipeline(pipeline)
    parallelism = (
        int(args.parallelism)
        if args.parallelism is not None
        else int(pipeline.get("parallelism", 4))
    )
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")

    configurations = _build_configurations(base, pipeline, Path("placeholder-policy.pt"))
    if args.dry_run:
        for config in configurations:
            config.validate()
        print(
            f"DRY RUN OK: {len(configurations)} configurations validated "
            f"with parallelism={parallelism}"
        )
        return 0

    repository = pipeline_path.parent.parent
    commit = _git_commit(repository) or "unknown"
    signature = _pipeline_signature(base, pipeline)
    output = (repository / pipeline["output_dir"]).resolve()
    compatible_commits = [
        str(value) for value in pipeline.get("compatible_resume_commits", [])
    ]
    session = _select_session(
        output,
        commit,
        signature,
        compatible_commits,
    )
    session.mkdir(parents=True, exist_ok=True)
    state_path = session / "pipeline-state.json"
    state = _load_state(
        state_path,
        commit,
        signature,
        compatible_commits,
    )
    state.setdefault("git_commit", commit)
    state["pipeline_signature"] = signature
    execution_commits = state.setdefault("execution_commits", [])
    if commit not in execution_commits:
        execution_commits.append(commit)

    training_rows_by_key: dict[str, dict] = {}
    checkpoints: dict[tuple[str, int], Path] = {}
    training_cases = [
        (strategy, int(vehicles))
        for vehicles in pipeline["vehicle_counts"]
        for strategy in pipeline["training_strategies"]
    ]
    pending_training: list[tuple[int, str, int, SimulationConfig]] = []
    for index, (strategy, vehicles) in enumerate(training_cases, start=1):
        key = f"train:{strategy}:{vehicles}"
        saved = state.get("runs", {}).get(key)
        if saved and Path(saved["checkpoint"]).exists():
            row = saved["row"]
            print(f"SKIP TRAIN {index}/{len(training_cases)} {strategy} vehicles={vehicles}", flush=True)
            training_rows_by_key[key] = row
            checkpoints[(strategy, vehicles)] = Path(saved["checkpoint"])
        else:
            config = _training_config(base, pipeline, strategy, vehicles, session)
            recovered = _recover_completed_run(config, "train")
            if recovered is not None:
                checkpoint = Path(recovered["run_dir"]) / "dqn-policy.pt"
                if checkpoint.exists():
                    print(
                        f"RECOVER TRAIN {index}/{len(training_cases)} "
                        f"{strategy} vehicles={vehicles}",
                        flush=True,
                    )
                    state.setdefault("runs", {})[key] = {
                        "checkpoint": str(checkpoint),
                        "row": recovered,
                    }
                    _write_json_atomic(state_path, state)
                    training_rows_by_key[key] = recovered
                    checkpoints[(strategy, vehicles)] = checkpoint
                    continue
            pending_training.append((index, strategy, vehicles, config))

    for index, strategy, vehicles, row in _execute_stage(
        pending_training,
        "TRAIN",
        len(training_cases),
        parallelism,
    ):
        key = f"train:{strategy}:{vehicles}"
        checkpoint = Path(row["run_dir"]) / "dqn-policy.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"training checkpoint was not created: {checkpoint}")
        state.setdefault("runs", {})[key] = {"checkpoint": str(checkpoint), "row": row}
        _write_json_atomic(state_path, state)
        print(
            f"DONE TRAIN {index}/{len(training_cases)} tasks={row['total_tasks']} "
            f"updates={row['dqn_updates']} wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )
        training_rows_by_key[key] = row
        checkpoints[(strategy, vehicles)] = checkpoint
    training_rows = [
        training_rows_by_key[f"train:{strategy}:{vehicles}"]
        for strategy, vehicles in training_cases
    ]

    evaluation_rows_by_key: dict[str, dict] = {}
    evaluation_cases = [
        (strategy, int(vehicles))
        for vehicles in pipeline["vehicle_counts"]
        for strategy in pipeline["evaluation_strategies"]
    ]
    pending_evaluation: list[tuple[int, str, int, SimulationConfig]] = []
    for index, (strategy, vehicles) in enumerate(evaluation_cases, start=1):
        key = f"evaluate:{strategy}:{vehicles}"
        saved = state.get("runs", {}).get(key)
        if saved and Path(saved["row"]["run_dir"], "summary.json").exists():
            row = saved["row"]
            print(f"SKIP EVAL {index}/{len(evaluation_cases)} {strategy} vehicles={vehicles}", flush=True)
            evaluation_rows_by_key[key] = row
        else:
            checkpoint = checkpoints.get(
                (_checkpoint_strategy(pipeline, strategy), vehicles)
            )
            config = _evaluation_config(base, pipeline, strategy, vehicles, checkpoint, session)
            recovered = _recover_completed_run(config, "evaluate")
            if recovered is not None:
                print(
                    f"RECOVER EVAL {index}/{len(evaluation_cases)} "
                    f"{strategy} vehicles={vehicles}",
                    flush=True,
                )
                state.setdefault("runs", {})[key] = {"row": recovered}
                _write_json_atomic(state_path, state)
                evaluation_rows_by_key[key] = recovered
                continue
            pending_evaluation.append((index, strategy, vehicles, config))

    for index, strategy, vehicles, row in _execute_stage(
        pending_evaluation,
        "EVAL",
        len(evaluation_cases),
        parallelism,
    ):
        key = f"evaluate:{strategy}:{vehicles}"
        state.setdefault("runs", {})[key] = {"row": row}
        _write_json_atomic(state_path, state)
        print(
            f"DONE EVAL {index}/{len(evaluation_cases)} success={100 * row['success_rate']:.2f}% "
            f"wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )
        evaluation_rows_by_key[key] = row
    evaluation_rows = [
        evaluation_rows_by_key[f"evaluate:{strategy}:{vehicles}"]
        for strategy, vehicles in evaluation_cases
    ]

    _write_csv(session / "training-results.csv", training_rows)
    _write_csv(session / "evaluation-results.csv", evaluation_rows)
    summary_paths = (
        session / "evaluation-diagnostics.csv",
        session / "diagnostic-summary.json",
        session / "diagnostic-summary.md",
    )
    if all(path.exists() for path in summary_paths):
        print(f"COMPLETE {session}")
        print(f"SUMMARY {session / 'diagnostic-summary.md'}")
        return 0
    diagnostics = _summarize_diagnostics(evaluation_rows)
    _write_csv(summary_paths[0], diagnostics["runs"])
    _write_json_atomic(summary_paths[1], diagnostics)
    summary_paths[2].write_text(
        _diagnostic_markdown(diagnostics), encoding="utf-8"
    )
    print(f"COMPLETE {session}")
    print(f"SUMMARY {session / 'diagnostic-summary.md'}")
    return 0


def _execute_stage(
    pending: list[tuple[int, str, int, SimulationConfig]],
    label: str,
    total: int,
    parallelism: int,
):
    if not pending:
        return
    _prepare_shared_scenarios(pending)
    _prepare_shared_mobility(pending)
    workers = min(parallelism, len(pending))
    if workers == 1:
        for index, strategy, vehicles, config in pending:
            print(
                f"START {label} {index}/{total} {strategy} vehicles={vehicles}",
                flush=True,
            )
            yield index, strategy, vehicles, _execute(
                config, "train" if label == "TRAIN" else "evaluate"
            )
        return
    scheduled = sorted(
        pending,
        key=lambda item: _estimated_run_cost(item[1], item[2]),
        reverse=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, strategy, vehicles, config in scheduled:
            print(
                f"START {label} {index}/{total} {strategy} vehicles={vehicles}",
                flush=True,
            )
            future = executor.submit(
                _execute,
                config,
                "train" if label == "TRAIN" else "evaluate",
            )
            futures[future] = (index, strategy, vehicles)
        failures = []
        for future in as_completed(futures):
            index, strategy, vehicles = futures[future]
            try:
                row = future.result()
            except Exception as error:
                failures.append((index, strategy, vehicles, error))
                print(
                    f"FAILED {label} {index}/{total} {strategy} "
                    f"vehicles={vehicles}: {error}",
                    flush=True,
                )
                continue
            yield index, strategy, vehicles, row
        if failures:
            index, strategy, vehicles, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} {label.lower()} run(s) failed; first failure "
                f"was {index}/{total} {strategy} vehicles={vehicles}"
            ) from error


def _estimated_run_cost(strategy: str, vehicles: int) -> float:
    """Order long cases first to reduce the four-worker phase makespan."""
    strategy_factor = {
        "random": 1.20,
        "hybrid_stackelberg": 1.12,
        "dqn": 1.10,
        "stackelberg": 1.00,
        "greedy": 0.98,
    }[strategy]
    return float(vehicles) ** 2 * strategy_factor


def _prepare_shared_scenarios(
    pending: list[tuple[int, str, int, SimulationConfig]],
) -> None:
    prepared: dict[tuple[str, str, int, float, int], str] = {}
    for _index, _strategy, _vehicles, config in pending:
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


def _prepare_shared_mobility(
    pending: list[tuple[int, str, int, SimulationConfig]],
) -> None:
    """Create each strategy-independent SUMO trace once before worker startup."""
    prepared: set[Path] = set()
    for _index, _strategy, vehicles, config in pending:
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
            f"PREPARE MOBILITY vehicles={vehicles} seed={config.seed}",
            flush=True,
        )
        mobility.start()
        try:
            for step in range(config.steps):
                mobility.step(step)
        finally:
            mobility.close()
        prepared.add(cache_path)


def _validate_pipeline(pipeline: dict) -> None:
    if min(int(pipeline["training_steps"]), int(pipeline["evaluation_steps"])) <= 0:
        raise ValueError("training_steps and evaluation_steps must be positive")
    if not pipeline["vehicle_counts"] or min(int(value) for value in pipeline["vehicle_counts"]) <= 0:
        raise ValueError("vehicle_counts must contain positive values")
    training = set(pipeline["training_strategies"])
    if not training or not training <= LEARNED_STRATEGIES:
        raise ValueError(
            "training_strategies must be a non-empty subset of "
            "dqn and hybrid_stackelberg"
        )
    unknown = set(pipeline["evaluation_strategies"]) - {
        "random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg"
    }
    if unknown:
        raise ValueError(f"unknown evaluation strategies: {sorted(unknown)}")
    hybrid_source = _hybrid_checkpoint_strategy(pipeline)
    if hybrid_source not in LEARNED_STRATEGIES:
        raise ValueError(
            "hybrid_checkpoint_strategy must be dqn or hybrid_stackelberg"
        )
    required_checkpoints = {
        _checkpoint_strategy(pipeline, strategy)
        for strategy in pipeline["evaluation_strategies"]
        if strategy in LEARNED_STRATEGIES
    }
    if not required_checkpoints <= training:
        raise ValueError(
            "every learned evaluation strategy must also be trained"
        )


def _hybrid_checkpoint_strategy(pipeline: dict) -> str:
    return str(pipeline.get("hybrid_checkpoint_strategy", "hybrid_stackelberg"))


def _checkpoint_strategy(pipeline: dict, strategy: str) -> str:
    """Return the training strategy whose checkpoint an evaluation loads.

    The decoupled Hybrid evaluates the game arbitration over the pure-DQN
    checkpoint, so its internal policy is never trained under the gate.
    """
    if strategy == "hybrid_stackelberg":
        return _hybrid_checkpoint_strategy(pipeline)
    return strategy


def _build_configurations(
    base: SimulationConfig, pipeline: dict, checkpoint: Path
) -> list[SimulationConfig]:
    configurations = []
    for vehicles in pipeline["vehicle_counts"]:
        for strategy in pipeline["training_strategies"]:
            configurations.append(
                _training_config(base, pipeline, strategy, int(vehicles), Path("dry-run"))
            )
        for strategy in pipeline["evaluation_strategies"]:
            selected = checkpoint if strategy in LEARNED_STRATEGIES else None
            configurations.append(
                _evaluation_config(
                    base, pipeline, strategy, int(vehicles), selected, Path("dry-run")
                )
            )
    return configurations


def _training_config(
    base: SimulationConfig, pipeline: dict, strategy: str, vehicles: int, session: Path
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = strategy
    config.vehicle_count = vehicles
    config.steps = int(pipeline["training_steps"])
    config.seed = int(pipeline["training_seed"])
    config.output_dir = str(session / "training" / f"{strategy}-{vehicles}")
    config.dqn.mode = "train"
    config.dqn.checkpoint_path = None
    config.record_decision_diagnostics = False
    training_sample_rate = float(
        pipeline.get("training_task_sample_rate", 0.0)
    )
    config.record_task_records = training_sample_rate > 0.0
    config.task_record_sample_rate = training_sample_rate
    config.validate()
    return config


def _evaluation_config(
    base: SimulationConfig,
    pipeline: dict,
    strategy: str,
    vehicles: int,
    checkpoint: Path | None,
    session: Path,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = strategy
    config.vehicle_count = vehicles
    config.steps = int(pipeline["evaluation_steps"])
    config.seed = int(pipeline["evaluation_seed"])
    config.output_dir = str(session / "evaluation" / f"{strategy}-{vehicles}")
    config.record_decision_diagnostics = True
    evaluation_sample_rate = float(
        pipeline.get("evaluation_task_sample_rate", 1.0)
    )
    config.record_task_records = evaluation_sample_rate > 0.0
    config.task_record_sample_rate = evaluation_sample_rate
    if strategy in LEARNED_STRATEGIES:
        if checkpoint is None:
            raise ValueError(f"evaluation checkpoint is required for {strategy}")
        config.dqn.mode = "evaluate"
        config.dqn.checkpoint_path = str(checkpoint)
    else:
        config.dqn.mode = "train"
        config.dqn.checkpoint_path = None
    config.validate()
    return config


def _execute(config: SimulationConfig, phase: str) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    wall_s = perf_counter() - wall_started
    cpu_s = process_time() - cpu_started
    timing_path = Path(run_dir) / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    return {
        "phase": phase,
        **asdict(summary),
        "wall_clock_s": wall_s,
        "process_cpu_s": cpu_s,
        **{f"phase_{name}_s": value for name, value in timing["phase_seconds"].items()},
        "run_dir": run_dir,
    }


def _recover_completed_run(
    config: SimulationConfig,
    phase: str,
) -> dict | None:
    """Recover a completed run whose future finished after an earlier failure."""
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
            **{
                f"phase_{name}_s": value
                for name, value in timing["phase_seconds"].items()
            },
            "run_dir": str(run_dir),
        }
    return None


def _summarize_diagnostics(evaluation_rows: list[dict]) -> dict:
    run_diagnostics = []
    task_paths_by_key: dict[tuple[str, int], Path] = {}
    for result in evaluation_rows:
        key = (result["strategy"], int(result["configured_vehicle_count"]))
        task_path = Path(result["run_dir"], "tasks.csv")
        task_paths_by_key[key] = task_path
        diagnostic = {
            "strategy": result["strategy"],
            "vehicle_count": result["configured_vehicle_count"],
            "tasks": 0,
            "success_rate": result["success_rate"],
            "oracle_success_rate": result["oracle_success_rate"],
            "oracle_gap": result["oracle_success_rate"] - result["success_rate"],
            "avg_latency_s": result["avg_latency_s"],
            "avg_energy_j": result["avg_energy_j"],
            "avg_cost_per_task": result["avg_cost_per_task"],
            "avg_reward": result["avg_reward"],
        }
        feasible_counts = {action: 0 for action in ("local", "v2v", "v2i")}
        value_stats = {
            column: _StreamingStats()
            for column in (
                "source_workload_s",
                "v2v_target_workload_s",
                "max_service_workload_s",
                "cloud_queue_length",
                "q_local",
                "q_v2v",
                "q_v2i",
            )
        }
        with task_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                diagnostic["tasks"] += 1
                deadline = _number(row["task_deadline_s"])
                for action in ("local", "v2v", "v2i"):
                    estimate = _number(row[f"{action}_estimate_s"])
                    feasible_counts[action] += int(
                        math.isfinite(estimate) and estimate <= deadline
                    )
                for column, stats in value_stats.items():
                    stats.add(_number(row.get(column, "")))
        task_count = max(diagnostic["tasks"], 1)
        for action in ("local", "v2v", "v2i"):
            diagnostic[f"{action}_deadline_feasible_rate"] = (
                feasible_counts[action] / task_count
            )
        for column in (
            "source_workload_s",
            "v2v_target_workload_s",
            "max_service_workload_s",
            "cloud_queue_length",
        ):
            diagnostic[f"{column}_mean"] = value_stats[column].mean
            diagnostic[f"{column}_max"] = value_stats[column].maximum
        for column in ("q_local", "q_v2v", "q_v2i"):
            diagnostic[f"{column}_mean"] = value_stats[column].mean
            diagnostic[f"{column}_std"] = value_stats[column].standard_deviation
        diagnostic["hybrid_deviation_ratio"] = result["hybrid_deviation_ratio"]
        diagnostic["hybrid_beneficial_deviation_rate"] = result[
            "hybrid_beneficial_deviation_rate"
        ]
        diagnostic["all_actions_late_rate"] = result.get("all_actions_late_rate", 0.0)
        diagnostic["all_late_cloud_admission_rate"] = result.get(
            "all_late_cloud_admission_rate", 0.0
        )
        diagnostic["all_late_cloud_to_capacity_ratio"] = result.get(
            "all_late_cloud_to_capacity_ratio", 0.0
        )
        diagnostic["dqn_deviation_ratio"] = result.get("dqn_deviation_ratio", 0.0)
        diagnostic["rule_deviation_ratio"] = result.get("rule_deviation_ratio", 0.0)
        run_diagnostics.append(diagnostic)

    comparisons = []
    vehicle_counts = sorted({int(row["configured_vehicle_count"]) for row in evaluation_rows})
    for vehicles in vehicle_counts:
        hybrid_path = task_paths_by_key.get(("hybrid_stackelberg", vehicles))
        stack_path = task_paths_by_key.get(("stackelberg", vehicles))
        if hybrid_path is None or stack_path is None:
            continue
        common = 0
        hybrid_wins = 0
        hybrid_losses = 0
        reward_delta = _StreamingStats()
        delay_delta = _StreamingStats()
        with (
            hybrid_path.open(encoding="utf-8", newline="") as hybrid_handle,
            stack_path.open(encoding="utf-8", newline="") as stack_handle,
        ):
            hybrid_reader = csv.DictReader(hybrid_handle)
            stack_reader = csv.DictReader(stack_handle)
            for hybrid, stack in zip(hybrid_reader, stack_reader, strict=True):
                if hybrid["task_id"] != stack["task_id"]:
                    raise RuntimeError(
                        "Hybrid and Stackelberg task streams are not aligned for "
                        f"{vehicles} vehicles: {hybrid['task_id']} != {stack['task_id']}"
                    )
                common += 1
                hybrid_wins += int(hybrid["success"]) > int(stack["success"])
                hybrid_losses += int(hybrid["success"]) < int(stack["success"])
                reward_delta.add(
                    _number(hybrid["reward"]) - _number(stack["reward"])
                )
                delay_delta.add(
                    _number(hybrid["delay_s"]) - _number(stack["delay_s"])
                )
        comparisons.append(
            {
                "vehicle_count": vehicles,
                "common_tasks": common,
                "hybrid_success_wins": hybrid_wins,
                "hybrid_success_losses": hybrid_losses,
                "hybrid_minus_stack_avg_reward": reward_delta.mean,
                "hybrid_minus_stack_avg_delay_s": delay_delta.mean,
            }
        )
    return {"runs": run_diagnostics, "hybrid_vs_stackelberg": comparisons}


class _StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squares = 0.0
        self.maximum = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_squares += value * value
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def standard_deviation(self) -> float:
        if self.count <= 1:
            return 0.0
        variance = self.total_squares / self.count - self.mean * self.mean
        return math.sqrt(max(variance, 0.0))


def _diagnostic_markdown(diagnostics: dict) -> str:
    lines = [
        "# Training and evaluation diagnostics",
        "",
        "Frozen evaluation results:",
        "",
        "| Strategy | Vehicles | Success | Oracle | Gap | Latency (s) | Energy (J) | Cost/task | Reward | Hybrid deviation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics["runs"]:
        lines.append(
            f"| {row['strategy']} | {row['vehicle_count']} | {100 * row['success_rate']:.2f}% | "
            f"{100 * row['oracle_success_rate']:.2f}% | {100 * row['oracle_gap']:.2f}% | "
            f"{row['avg_latency_s']:.4f} | {row['avg_energy_j']:.2f} | "
            f"{row['avg_cost_per_task']:.4f} | {row['avg_reward']:.4f} | "
            f"{100 * row['hybrid_deviation_ratio']:.2f}% |"
        )
    lines.extend([
        "",
        "Action feasibility and queue/Q diagnostics are preserved in `evaluation-diagnostics.csv` and `diagnostic-summary.json`.",
    ])
    lines.extend(["", "Hybrid versus Stackelberg on identical task IDs:", ""])
    for row in diagnostics["hybrid_vs_stackelberg"]:
        lines.append(
            f"- {row['vehicle_count']} vehicles: {row['hybrid_success_wins']} success wins, "
            f"{row['hybrid_success_losses']} losses, average reward delta "
            f"{row['hybrid_minus_stack_avg_reward']:.4f}, average delay delta "
            f"{row['hybrid_minus_stack_avg_delay_s']:.6f} s."
        )
    return "\n".join(lines) + "\n"


def _number(value: str | int | float | None) -> float:
    if value in (None, ""):
        return math.inf
    return float(value)


def _pipeline_signature(base: SimulationConfig, pipeline: dict) -> str:
    stable_pipeline = {
        key: value
        for key, value in pipeline.items()
        if key != "compatible_resume_commits"
    }
    encoded = json.dumps(
        {"base": base.to_dict(), "pipeline": stable_pipeline},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


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


def _select_session(
    output: Path,
    commit: str,
    signature: str,
    compatible_commits: list[str],
) -> Path:
    current = output / f"run-{commit[:8]}-{signature}"
    if current.exists():
        return current
    for compatible in compatible_commits:
        candidate = output / f"run-{compatible[:8]}-{signature}"
        if candidate.exists():
            return candidate
    return current


def _load_state(
    path: Path,
    commit: str,
    signature: str,
    compatible_commits: list[str] | None = None,
) -> dict:
    if not path.exists():
        return {"runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    allowed_commits = {commit, *(compatible_commits or [])}
    if (
        state.get("git_commit") not in allowed_commits
        or state.get("pipeline_signature") != signature
    ):
        raise RuntimeError("pipeline state does not match the current commit and configuration")
    return state


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
