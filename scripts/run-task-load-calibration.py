from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.simulation import SimulationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate task load and validate Hybrid delegation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    calibration = raw["calibration"]
    repository = config_path.parent.parent
    base = SimulationConfig.from_toml((config_path.parent / calibration["base_config"]).resolve())
    _validate_base(base)
    _validate_calibration(calibration, base)

    if args.dry_run:
        _dry_run(base, calibration)
        return 0

    commit = _git_commit(repository) or "unknown"
    signature = sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    session = (
        repository / calibration["output_dir"] / f"run-{commit[:8]}-{signature}"
    ).resolve()
    session.mkdir(parents=True, exist_ok=True)
    state_path = session / "calibration-state.json"
    state = _load_state(state_path, commit, signature)
    state.update({"git_commit": commit, "signature": signature})

    probabilities = [float(value) for value in calibration["probabilities"]]
    screening_rows = []
    print(
        f"PHASE 1/3 load calibration: {len(probabilities)} Stackelberg runs at "
        f"{calibration['screen_vehicle_count']} vehicles.",
        flush=True,
    )
    for index, probability in enumerate(probabilities, start=1):
        key = f"screen:{probability:.6f}"
        row = _saved_row(state, key)
        if row is not None:
            print(f"SKIP SCREEN {index}/{len(probabilities)} p={probability:.2f}", flush=True)
        else:
            config = _base_case(
                base,
                probability,
                int(calibration["screen_vehicle_count"]),
                int(calibration["screening_steps"]),
                int(calibration["screening_seed"]),
                "stackelberg",
                session / "screening" / f"p-{probability:.2f}",
                record_tasks=False,
            )
            print(f"START SCREEN {index}/{len(probabilities)} p={probability:.2f}", flush=True)
            row = _execute(config, "screen", probability)
            _save_row(state, state_path, key, row)
            print(
                f"DONE SCREEN {index}/{len(probabilities)} p={probability:.2f} "
                f"offered={row['offered_vehicle_compute_load_ratio']:.3f} "
                f"queue-timeout={100 * row['queue_induced_local_timeout_ratio']:.2f}% "
                f"v2v-rescue={100 * row['v2v_rescuable_task_ratio']:.2f}% "
                f"success={100 * row['success_rate']:.2f}% wall={row['wall_clock_s']:.1f}s",
                flush=True,
            )
        screening_rows.append(row)

    target = float(calibration["target_offered_vehicle_compute_load_ratio"])
    selected = min(
        screening_rows,
        key=lambda row: (
            abs(float(row["offered_vehicle_compute_load_ratio"]) - target),
            float(row["task_probability"]),
        ),
    )
    probability = float(selected["task_probability"])
    selection = {
        "selected_probability": probability,
        "target_offered_vehicle_compute_load_ratio": target,
        "observed_offered_vehicle_compute_load_ratio": float(
            selected["offered_vehicle_compute_load_ratio"]
        ),
        "selection_rule": (
            "minimum absolute distance to the predeclared vehicle compute load target; "
            "success and Hybrid advantage are excluded"
        ),
    }
    _write_json_atomic(session / "selected-load.json", selection)
    print(
        f"SELECT p={probability:.2f}: offered="
        f"{selection['observed_offered_vehicle_compute_load_ratio']:.3f} "
        f"closest to target={target:.3f}.",
        flush=True,
    )

    bandwidths = [
        float(value) for value in calibration["v2v_channel_bandwidths_mhz"]
    ]
    bandwidth_rows = []
    print(
        f"PHASE 2/3 V2V opportunity calibration: {len(bandwidths)} bandwidths at "
        f"p={probability:.2f}.",
        flush=True,
    )
    for index, bandwidth in enumerate(bandwidths, start=1):
        if bandwidth == base.network.v2v_channel_bandwidth_mhz:
            row = dict(selected)
            row["phase"] = "bandwidth_screen"
            print(
                f"REUSE CHANNEL {index}/{len(bandwidths)} B={bandwidth:.0f} MHz",
                flush=True,
            )
        else:
            key = f"bandwidth:{probability:.6f}:{bandwidth:.6f}"
            row = _saved_row(state, key)
            if row is not None:
                print(
                    f"SKIP CHANNEL {index}/{len(bandwidths)} B={bandwidth:.0f} MHz",
                    flush=True,
                )
            else:
                config = _base_case(
                    base,
                    probability,
                    int(calibration["screen_vehicle_count"]),
                    int(calibration["screening_steps"]),
                    int(calibration["screening_seed"]),
                    "stackelberg",
                    session / "bandwidth-screening" / f"b-{bandwidth:.0f}",
                    record_tasks=False,
                    channel_bandwidth_mhz=bandwidth,
                )
                print(
                    f"START CHANNEL {index}/{len(bandwidths)} B={bandwidth:.0f} MHz",
                    flush=True,
                )
                row = _execute(config, "bandwidth_screen", probability)
                _save_row(state, state_path, key, row)
                print(
                    f"DONE CHANNEL {index}/{len(bandwidths)} B={bandwidth:.0f} MHz "
                    f"faster={100 * row['v2v_latency_advantage_ratio']:.2f}% "
                    f"rescue={100 * row['v2v_rescuable_task_ratio']:.2f}% "
                    f"wall={row['wall_clock_s']:.1f}s",
                    flush=True,
                )
        bandwidth_rows.append(row)

    bandwidth_target = float(calibration["target_v2v_latency_advantage_ratio"])
    selected_bandwidth_row = min(
        bandwidth_rows,
        key=lambda row: (
            abs(float(row["v2v_latency_advantage_ratio"]) - bandwidth_target),
            float(row["v2v_channel_bandwidth_mhz"]),
        ),
    )
    bandwidth = float(selected_bandwidth_row["v2v_channel_bandwidth_mhz"])
    selection.update(
        {
            "selected_v2v_channel_bandwidth_mhz": bandwidth,
            "target_v2v_latency_advantage_ratio": bandwidth_target,
            "observed_v2v_latency_advantage_ratio": float(
                selected_bandwidth_row["v2v_latency_advantage_ratio"]
            ),
            "bandwidth_selection_rule": (
                "minimum absolute distance to the predeclared V2V latency-advantage "
                "target; success and Hybrid advantage are excluded"
            ),
        }
    )
    _write_json_atomic(session / "selected-load.json", selection)
    print(
        f"SELECT B={bandwidth:.0f} MHz: V2V faster="
        f"{100 * selection['observed_v2v_latency_advantage_ratio']:.2f}% "
        f"versus target={100 * bandwidth_target:.2f}%.",
        flush=True,
    )

    vehicle_counts = [int(value) for value in calibration["vehicle_counts"]]
    checkpoints: dict[int, Path] = {}
    training_rows = []
    print(
        f"PHASE 3/3 train and validate p={probability:.2f}, B={bandwidth:.0f} MHz "
        f"across {vehicle_counts}.",
        flush=True,
    )
    for index, vehicles in enumerate(vehicle_counts, start=1):
        key = f"train:hybrid:{probability:.6f}:{bandwidth:.6f}:{vehicles}"
        row = _saved_row(state, key)
        checkpoint = Path(row["run_dir"]) / "dqn-policy.pt" if row else None
        if row is not None and checkpoint is not None and checkpoint.exists():
            print(f"SKIP TRAIN {index}/{len(vehicle_counts)} vehicles={vehicles}", flush=True)
        else:
            config = _base_case(
                base,
                probability,
                vehicles,
                int(calibration["training_steps"]),
                int(calibration["training_seed"]),
                "hybrid_stackelberg",
                session / "training" / f"hybrid-{vehicles}",
                record_tasks=False,
                channel_bandwidth_mhz=bandwidth,
            )
            config.dqn.mode = "train"
            config.dqn.checkpoint_path = None
            config.validate()
            print(f"START TRAIN {index}/{len(vehicle_counts)} vehicles={vehicles}", flush=True)
            row = _execute(config, "train", probability)
            checkpoint = Path(row["run_dir"]) / "dqn-policy.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"training checkpoint not created: {checkpoint}")
            _save_row(state, state_path, key, row)
            print(
                f"DONE TRAIN {index}/{len(vehicle_counts)} tasks={row['total_tasks']} "
                f"updates={row['dqn_updates']} wall={row['wall_clock_s']:.1f}s",
                flush=True,
            )
        training_rows.append(row)
        checkpoints[vehicles] = checkpoint

    evaluation_rows = []
    evaluation_cases = [
        (vehicles, strategy)
        for vehicles in vehicle_counts
        for strategy in ("stackelberg", "hybrid_stackelberg")
    ]
    for index, (vehicles, strategy) in enumerate(evaluation_cases, start=1):
        key = f"evaluate:{strategy}:{probability:.6f}:{bandwidth:.6f}:{vehicles}"
        row = _saved_row(state, key, require_tasks=True)
        if row is not None:
            print(
                f"SKIP EVAL {index}/{len(evaluation_cases)} {strategy} vehicles={vehicles}",
                flush=True,
            )
        else:
            config = _base_case(
                base,
                probability,
                vehicles,
                int(calibration["evaluation_steps"]),
                int(calibration["evaluation_seed"]),
                strategy,
                session / "evaluation" / f"{strategy}-{vehicles}",
                record_tasks=True,
                channel_bandwidth_mhz=bandwidth,
            )
            if strategy == "hybrid_stackelberg":
                config.dqn.mode = "evaluate"
                config.dqn.checkpoint_path = str(checkpoints[vehicles])
            else:
                config.dqn.mode = "train"
                config.dqn.checkpoint_path = None
            config.validate()
            print(
                f"START EVAL {index}/{len(evaluation_cases)} {strategy} vehicles={vehicles}",
                flush=True,
            )
            row = _execute(config, "evaluate", probability)
            _save_row(state, state_path, key, row)
            print(
                f"DONE EVAL {index}/{len(evaluation_cases)} {strategy} vehicles={vehicles} "
                f"success={100 * row['success_rate']:.2f}% queue={row['avg_cloud_queue_length']:.2f} "
                f"wall={row['wall_clock_s']:.1f}s",
                flush=True,
            )
        evaluation_rows.append(row)

    comparisons = _comparisons(evaluation_rows, vehicle_counts)
    _write_csv(session / "load-screening.csv", screening_rows)
    _write_csv(session / "bandwidth-screening.csv", bandwidth_rows)
    _write_csv(session / "training-results.csv", training_rows)
    _write_csv(session / "evaluation-results.csv", evaluation_rows)
    _write_csv(session / "hybrid-vs-stackelberg.csv", comparisons)
    (session / "calibration-summary.md").write_text(
        _markdown(screening_rows, bandwidth_rows, selection, evaluation_rows, comparisons),
        encoding="utf-8",
    )
    print(f"COMPLETE {session}")
    print(f"SUMMARY {session / 'calibration-summary.md'}")
    return 0


def _validate_base(base: SimulationConfig) -> None:
    if base.service_role_mode != "dynamic_idle":
        raise ValueError("load calibration requires thesis dynamic-idle service roles")
    if not base.decision.stackelberg_deadline_action_masking:
        raise ValueError("reviewed calibration requires on-time-first Stackelberg decisions")
    if base.service_compute_hz != base.vehicle_compute_hz:
        raise ValueError("thesis dynamic-idle vehicles must use one homogeneous CPU rate")
    if base.task_deadline_distribution != "uniform":
        raise ValueError("homogeneous load calibration requires continuous task deadlines")


def _validate_calibration(calibration: dict, base: SimulationConfig) -> None:
    bandwidths = [
        float(value) for value in calibration["v2v_channel_bandwidths_mhz"]
    ]
    if not bandwidths or min(bandwidths) <= 0 or len(set(bandwidths)) != len(bandwidths):
        raise ValueError("V2V bandwidth candidates must be unique positive values")
    if base.network.v2v_channel_bandwidth_mhz not in bandwidths:
        raise ValueError("V2V bandwidth candidates must include the base bandwidth")
    target = float(calibration["target_v2v_latency_advantage_ratio"])
    if not 0.0 <= target <= 1.0:
        raise ValueError("V2V latency-advantage target must be in [0, 1]")


def _dry_run(base: SimulationConfig, calibration: dict) -> None:
    probabilities = [float(value) for value in calibration["probabilities"]]
    for probability in probabilities:
        _base_case(
            base,
            probability,
            int(calibration["screen_vehicle_count"]),
            int(calibration["screening_steps"]),
            int(calibration["screening_seed"]),
            "stackelberg",
            Path("dry-run"),
            False,
        ).validate()
    selected = probabilities[0]
    bandwidths = [
        float(value) for value in calibration["v2v_channel_bandwidths_mhz"]
    ]
    for bandwidth in bandwidths:
        _base_case(
            base,
            selected,
            int(calibration["screen_vehicle_count"]),
            int(calibration["screening_steps"]),
            int(calibration["screening_seed"]),
            "stackelberg",
            Path("dry-run"),
            False,
            bandwidth,
        ).validate()
    selected_bandwidth = bandwidths[0]
    for vehicles in calibration["vehicle_counts"]:
        for strategy in ("stackelberg", "hybrid_stackelberg"):
            config = _base_case(
                base,
                selected,
                int(vehicles),
                int(calibration["evaluation_steps"]),
                int(calibration["evaluation_seed"]),
                strategy,
                Path("dry-run"),
                True,
                selected_bandwidth,
            )
            if strategy == "hybrid_stackelberg":
                config.dqn.mode = "evaluate"
                config.dqn.checkpoint_path = "dry-run-policy.pt"
            config.validate()
    total = len(probabilities) + len(bandwidths) + 3 * len(calibration["vehicle_counts"])
    print(f"DRY RUN OK: {total} staged configurations validated")


def _base_case(
    base: SimulationConfig,
    probability: float,
    vehicles: int,
    steps: int,
    seed: int,
    strategy: str,
    output: Path,
    record_tasks: bool,
    channel_bandwidth_mhz: float | None = None,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.task_probability = probability
    config.vehicle_count = vehicles
    config.steps = steps
    config.seed = seed
    config.strategy = strategy
    config.output_dir = str(output)
    config.record_task_records = record_tasks
    config.record_decision_diagnostics = False
    if channel_bandwidth_mhz is not None:
        config.network.v2v_channel_bandwidth_mhz = channel_bandwidth_mhz
    config.validate()
    return config


def _execute(config: SimulationConfig, phase: str, probability: float) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    return {
        "phase": phase,
        "task_probability": probability,
        "v2v_channel_bandwidth_mhz": config.network.v2v_channel_bandwidth_mhz,
        "theoretical_offered_vehicle_compute_load_ratio": (
            probability * _mean_task_cycles(config) / config.vehicle_compute_hz
        ),
        **asdict(summary),
        "wall_clock_s": perf_counter() - wall_started,
        "process_cpu_s": process_time() - cpu_started,
        "run_dir": run_dir,
    }


def _mean_task_cycles(config: SimulationConfig) -> float:
    if config.task_compute_distribution == "uniform":
        return (config.task_compute_min_cycles + config.task_compute_max_cycles) / 2.0
    return sum(config.task_compute_choices) / len(config.task_compute_choices)


def _comparisons(rows: list[dict], vehicle_counts: list[int]) -> list[dict]:
    by_key = {
        (row["strategy"], int(row["configured_vehicle_count"])): row for row in rows
    }
    comparisons = []
    for vehicles in vehicle_counts:
        stack = by_key[("stackelberg", vehicles)]
        hybrid = by_key[("hybrid_stackelberg", vehicles)]
        wins = losses = changes = common = 0
        stack_path = Path(stack["run_dir"]) / "tasks.csv"
        hybrid_path = Path(hybrid["run_dir"]) / "tasks.csv"
        with stack_path.open(encoding="utf-8", newline="") as left, hybrid_path.open(
            encoding="utf-8", newline=""
        ) as right:
            for stack_task, hybrid_task in zip(
                csv.DictReader(left), csv.DictReader(right), strict=True
            ):
                if stack_task["task_id"] != hybrid_task["task_id"]:
                    raise RuntimeError("evaluation task streams are not identical")
                common += 1
                stack_success = int(stack_task["success"])
                hybrid_success = int(hybrid_task["success"])
                wins += hybrid_success > stack_success
                losses += hybrid_success < stack_success
                changes += hybrid_task["action"] != stack_task["action"]
        comparisons.append(
            {
                "vehicle_count": vehicles,
                "common_tasks": common,
                "hybrid_success_wins": wins,
                "hybrid_success_losses": losses,
                "hybrid_net_successes": wins - losses,
                "action_changes": changes,
                "action_change_ratio": changes / max(common, 1),
                "hybrid_minus_stack_success_pp": 100.0
                * (float(hybrid["success_rate"]) - float(stack["success_rate"])),
            }
        )
    return comparisons


def _markdown(
    screening: list[dict],
    bandwidth_screening: list[dict],
    selection: dict,
    evaluation: list[dict],
    comparisons: list[dict],
) -> str:
    lines = [
        "# Homogeneous-vehicle task-load calibration",
        "",
        "Every task and service vehicle uses the thesis-specified 2 GHz CPU. The task probability is selected only by distance to the predeclared offered vehicle-compute load target; success and Hybrid advantage are not selection inputs.",
        "",
        "| Probability | Theoretical load | Realized load | Source queue p95 | Queue timeout | V2V faster | V2V rescue | Success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screening:
        lines.append(
            f"| {float(row['task_probability']):.2f} | "
            f"{float(row['theoretical_offered_vehicle_compute_load_ratio']):.3f} | "
            f"{float(row['offered_vehicle_compute_load_ratio']):.3f} | "
            f"{float(row['p95_source_workload_s']):.3f}s | "
            f"{100 * float(row['queue_induced_local_timeout_ratio']):.2f}% | "
            f"{100 * float(row['v2v_latency_advantage_ratio']):.2f}% | "
            f"{100 * float(row['v2v_rescuable_task_ratio']):.2f}% | "
            f"{100 * float(row['success_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected probability: **{selection['selected_probability']:.2f}**, observed offered load "
            f"{selection['observed_offered_vehicle_compute_load_ratio']:.3f} versus target "
            f"{selection['target_offered_vehicle_compute_load_ratio']:.3f}.",
            "",
            "V2V bandwidth is selected by latency-opportunity distance, not success:",
            "",
            "| B0 | V2V faster | V2V rescue | V2V action | Success (diagnostic only) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bandwidth_screening:
        lines.append(
            f"| {float(row['v2v_channel_bandwidth_mhz']):.0f} MHz | "
            f"{100 * float(row['v2v_latency_advantage_ratio']):.2f}% | "
            f"{100 * float(row['v2v_rescuable_task_ratio']):.2f}% | "
            f"{100 * float(row['v2v_offload_ratio']):.2f}% | "
            f"{100 * float(row['success_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected B: **{selection['selected_v2v_channel_bandwidth_mhz']:.0f} MHz**, observed "
            f"V2V latency advantage {100 * selection['observed_v2v_latency_advantage_ratio']:.2f}% "
            f"versus target {100 * selection['target_v2v_latency_advantage_ratio']:.2f}%.",
            "",
            "| Vehicles | Strategy | Success | Reward | Queue | DQN decisions | Deviations |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation:
        lines.append(
            f"| {row['configured_vehicle_count']} | {row['strategy']} | "
            f"{100 * float(row['success_rate']):.2f}% | {float(row['avg_reward']):.3f} | "
            f"{float(row['avg_cloud_queue_length']):.2f} | "
            f"{100 * float(row['dqn_decision_ratio']):.2f}% | "
            f"{100 * float(row['hybrid_deviation_ratio']):.2f}% |"
        )
    lines.extend(["", "Pairwise Hybrid versus Stackelberg:", ""])
    for row in comparisons:
        lines.append(
            f"- {row['vehicle_count']} vehicles: {row['hybrid_success_wins']} wins, "
            f"{row['hybrid_success_losses']} losses, net {row['hybrid_net_successes']}, "
            f"success delta {row['hybrid_minus_stack_success_pp']:+.3f} pp, "
            f"action changes {100 * row['action_change_ratio']:.2f}%."
        )
    return "\n".join(lines) + "\n"


def _saved_row(state: dict, key: str, require_tasks: bool = False) -> dict | None:
    saved = state.get("runs", {}).get(key)
    if not saved:
        return None
    row = saved["row"]
    run_dir = Path(row["run_dir"])
    if not (run_dir / "summary.json").exists():
        return None
    if require_tasks and not (run_dir / "tasks.csv").exists():
        return None
    return row


def _save_row(state: dict, state_path: Path, key: str, row: dict) -> None:
    state.setdefault("runs", {})[key] = {"row": row}
    _write_json_atomic(state_path, state)


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


def _load_state(path: Path, commit: str, signature: str) -> dict:
    if not path.exists():
        return {"runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("git_commit") != commit or state.get("signature") != signature:
        raise RuntimeError("calibration state does not match the current commit and configuration")
    return state


def _write_csv(path: Path, rows: list[dict]) -> None:
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
