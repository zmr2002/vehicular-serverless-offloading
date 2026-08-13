from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from vehicular_offloading.config import SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the staged Hybrid causal, RL, confirmation, and final study"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hybrid-optimization-study.toml"),
    )
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    repository = _repository_root(config_path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    study = dict(raw["study"])
    causal_profiles = raw["causal_profiles"]
    rl_profiles = raw["rl_profiles"]
    parallelism = int(args.parallelism or study["parallelism"])
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")

    if args.smoke:
        study.update(
            {
                "output_dir": "results/verified/hybrid-optimization-smoke",
                "steps": 50,
                "screen_vehicle_count": 100,
                "confirmation_vehicle_counts": [100],
                "final_vehicle_counts": [100],
                "causal_training_seeds": [32601, 32602],
                "causal_evaluation_seeds": [91, 92],
                "rl_training_seeds": [32611, 32612],
                "rl_evaluation_seeds": [93, 94],
                "confirmation_training_seeds": [32621, 32622],
                "confirmation_evaluation_seeds": [95, 96],
                "final_training_seeds": [32631, 32632],
                "final_evaluation_seeds": [97, 98],
                "serverless_enabled": False,
            }
        )
        causal_profiles = {
            key: causal_profiles[key]
            for key in ("current", "adaptive_only", "legacy_retrained", "fixed_point")
        }
        rl_profiles = {
            key: rl_profiles[key]
            for key in ("base", "gamma_097", "gamma_097_replay_100k")
        }

    output = _resolve(repository, Path(study["output_dir"]))
    generated = output / "generated-configs"
    generated.mkdir(parents=True, exist_ok=True)
    _check_free_disk(output, float(study["minimum_free_disk_gb"]))

    profile_paths = _generate_causal_profiles(
        repository,
        generated,
        causal_profiles,
    )
    if args.dry_run:
        _dry_run_profiles(
            repository,
            generated,
            profile_paths,
            rl_profiles,
            study,
            parallelism,
        )
        print("DRY RUN COMPLETE: all profiles and staged pipeline shapes validated.")
        return 0

    state_path = output / "study-state.json"
    state = _load_state(state_path)
    state.update(
        {
            "status": "running",
            "config": str(config_path),
            "output_dir": str(output),
            "parallelism": parallelism,
            "smoke": bool(args.smoke),
            "started_at": state.get("started_at") or _utc_now(),
        }
    )
    _write_json(state_path, state)

    print("PHASE 1/5: causal attribution and configuration screen.", flush=True)
    causal_results = {}
    for index, (name, profile_path) in enumerate(profile_paths.items(), start=1):
        print(f"CAUSAL {index}/{len(profile_paths)} {name}", flush=True)
        result_root = output / "causal" / name
        _run_multiseed(
            repository,
            generated / f"causal-{name}-pipeline.toml",
            profile_path,
            result_root,
            study,
            study["causal_training_seeds"],
            study["causal_evaluation_seeds"],
            [int(study["screen_vehicle_count"])],
            ["hybrid_stackelberg"],
            ["hybrid_stackelberg"],
            parallelism,
        )
        causal_results[name] = _hybrid_screen_metrics(result_root)
        _check_free_disk(output, float(study["minimum_free_disk_gb"]))
    causal_winner = _select_screen_winner(
        causal_results,
        float(study["screen_min_material_gain"]),
    )
    _write_selection(output / "causal-selection.csv", causal_results, causal_winner)
    print(f"SELECT CAUSAL {causal_winner}", flush=True)

    print("PHASE 2/5: long-horizon DQN screen on the selected causal model.", flush=True)
    rl_profile_paths = _generate_rl_profiles(
        generated,
        profile_paths[causal_winner],
        rl_profiles,
    )
    rl_results = {}
    for index, (name, profile_path) in enumerate(rl_profile_paths.items(), start=1):
        print(f"RL {index}/{len(rl_profile_paths)} {name}", flush=True)
        result_root = output / "rl" / name
        _run_multiseed(
            repository,
            generated / f"rl-{name}-pipeline.toml",
            profile_path,
            result_root,
            study,
            study["rl_training_seeds"],
            study["rl_evaluation_seeds"],
            [int(study["screen_vehicle_count"])],
            ["hybrid_stackelberg"],
            ["hybrid_stackelberg"],
            parallelism,
        )
        rl_results[name] = _hybrid_screen_metrics(result_root)
        _check_free_disk(output, float(study["minimum_free_disk_gb"]))
    rl_winner = _select_screen_winner(
        rl_results,
        float(study["screen_min_material_gain"]),
    )
    _write_selection(output / "rl-selection.csv", rl_results, rl_winner)
    print(f"SELECT RL {rl_winner}", flush=True)

    print("PHASE 3/5: paired confirmation at 2000 and 4000 vehicles.", flush=True)
    confirmation_candidates = {
        "current": profile_paths["current"],
        "legacy_retrained": profile_paths["legacy_retrained"],
    }
    for name in _rank_screen_profiles(causal_results)[
        : int(study["causal_confirmation_top"])
    ]:
        confirmation_candidates[f"causal_{name}"] = profile_paths[name]
    for name in _rank_screen_profiles(rl_results)[
        : int(study["rl_confirmation_top"])
    ]:
        confirmation_candidates[f"enhanced_{causal_winner}_{name}"] = (
            rl_profile_paths[name]
        )
    confirmation_profiles = _unique_profiles(confirmation_candidates)
    confirmation_results = {}
    for index, (name, profile_path) in enumerate(confirmation_profiles.items(), start=1):
        print(f"CONFIRM {index}/{len(confirmation_profiles)} {name}", flush=True)
        result_root = output / "confirmation" / name
        _run_multiseed(
            repository,
            generated / f"confirmation-{name}-pipeline.toml",
            profile_path,
            result_root,
            study,
            study["confirmation_training_seeds"],
            study["confirmation_evaluation_seeds"],
            [int(value) for value in study["confirmation_vehicle_counts"]],
            ["dqn", "hybrid_stackelberg"],
            ["dqn", "stackelberg", "hybrid_stackelberg"],
            parallelism,
        )
        confirmation_results[name] = _confirmation_metrics(result_root)
        _check_free_disk(output, float(study["minimum_free_disk_gb"]))
    final_profile_name = _select_confirmation_winner(
        confirmation_results,
        float(study["confirmation_min_material_gain"]),
    )
    final_profile = confirmation_profiles[final_profile_name]
    _write_confirmation_selection(
        output / "confirmation-selection.csv",
        confirmation_results,
        final_profile_name,
    )
    print(f"SELECT FINAL PROFILE {final_profile_name}", flush=True)

    print("PHASE 4/5: independent final five-strategy, three-scale matrix.", flush=True)
    final_root = output / "final"
    final_pipeline = generated / "final-pipeline.toml"
    _run_multiseed(
        repository,
        final_pipeline,
        final_profile,
        final_root,
        study,
        study["final_training_seeds"],
        study["final_evaluation_seeds"],
        [int(value) for value in study["final_vehicle_counts"]],
        ["dqn", "hybrid_stackelberg"],
        ["random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg"],
        parallelism,
    )
    final_metrics = _confirmation_metrics(final_root)
    serverless_ready, gate = _serverless_gate(
        final_metrics,
        confirmation_results,
        final_profile_name,
        float(study["serverless_success_tolerance"]),
        float(study["serverless_current_regression_tolerance"]),
    )
    serverless_ready = serverless_ready and bool(study["serverless_enabled"])

    state.update(
        {
            "status": "analytical_complete",
            "completed_at": _utc_now(),
            "causal_winner": causal_winner,
            "rl_winner": rl_winner,
            "final_profile_name": final_profile_name,
            "final_profile": str(final_profile),
            "final_pipeline": str(final_pipeline),
            "final_output": str(final_root),
            "serverless_ready": serverless_ready,
            "serverless_gate": gate,
        }
    )
    _write_json(state_path, state)
    _write_study_summary(
        output / "study-summary.md",
        causal_results,
        causal_winner,
        rl_results,
        rl_winner,
        confirmation_results,
        final_profile_name,
        final_metrics,
        gate,
        serverless_ready,
    )
    print("PHASE 5/5: analytical study complete.", flush=True)
    print(f"STATE {state_path}", flush=True)
    print(f"SUMMARY {output / 'study-summary.md'}", flush=True)
    print(f"SERVERLESS_READY {str(serverless_ready).lower()}", flush=True)
    return 0


def _dry_run_profiles(
    repository: Path,
    generated: Path,
    profiles: dict[str, Path],
    rl_profiles: dict,
    study: dict,
    parallelism: int,
) -> None:
    for name, profile in profiles.items():
        _run_multiseed(
            repository,
            generated / f"dry-{name}.toml",
            profile,
            generated / "dry-output" / name,
            study,
            study["causal_training_seeds"],
            study["causal_evaluation_seeds"],
            [int(study["screen_vehicle_count"])],
            ["hybrid_stackelberg"],
            ["hybrid_stackelberg"],
            parallelism,
            dry_run=True,
        )
    generated_rl = _generate_rl_profiles(generated, profiles["current"], rl_profiles)
    for name, profile in generated_rl.items():
        _run_multiseed(
            repository,
            generated / f"dry-rl-{name}.toml",
            profile,
            generated / "dry-output" / "rl" / name,
            study,
            study["rl_training_seeds"],
            study["rl_evaluation_seeds"],
            [int(study["screen_vehicle_count"])],
            ["hybrid_stackelberg"],
            ["hybrid_stackelberg"],
            parallelism,
            dry_run=True,
        )
    _run_multiseed(
        repository,
        generated / "dry-confirmation.toml",
        profiles["current"],
        generated / "dry-output" / "confirmation",
        study,
        study["confirmation_training_seeds"],
        study["confirmation_evaluation_seeds"],
        [int(value) for value in study["confirmation_vehicle_counts"]],
        ["dqn", "hybrid_stackelberg"],
        ["dqn", "stackelberg", "hybrid_stackelberg"],
        parallelism,
        dry_run=True,
    )
    _run_multiseed(
        repository,
        generated / "dry-final.toml",
        profiles["current"],
        generated / "dry-output" / "final",
        study,
        study["final_training_seeds"],
        study["final_evaluation_seeds"],
        [int(value) for value in study["final_vehicle_counts"]],
        ["dqn", "hybrid_stackelberg"],
        ["random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg"],
        parallelism,
        dry_run=True,
    )


def _generate_causal_profiles(
    repository: Path,
    generated: Path,
    profiles: dict,
) -> dict[str, Path]:
    result = {}
    for name, definition in profiles.items():
        definition = dict(definition)
        base = (repository / "configs" / definition.pop("base_config")).resolve()
        if not base.is_file():
            raise FileNotFoundError(f"base config not found: {base}")
        simulation = dict(definition.get("simulation", {}))
        simulation["record_decision_diagnostics"] = True
        definition["simulation"] = simulation
        path = generated / f"causal-{name}.toml"
        path.write_text(_render_overlay(base, definition), encoding="utf-8")
        result[name] = path
    required = {"current", "legacy_retrained"}
    if not required <= set(result):
        raise ValueError(f"causal profiles must include {sorted(required)}")
    return result


def _generate_rl_profiles(
    generated: Path,
    causal_profile: Path,
    profiles: dict,
) -> dict[str, Path]:
    result = {}
    for name, definition in profiles.items():
        path = generated / f"rl-{name}.toml"
        path.write_text(
            _render_overlay(causal_profile.resolve(), dict(definition)),
            encoding="utf-8",
        )
        result[name] = path
    return result


def _render_overlay(base: Path, sections: dict) -> str:
    lines = [f"extends = {json.dumps(base.as_posix())}", ""]
    for section_name, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"profile section {section_name} must be a table")
        lines.append(f"[{section_name}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value)


def _run_multiseed(
    repository: Path,
    pipeline_path: Path,
    profile_path: Path,
    output: Path,
    study: dict,
    training_seeds,
    evaluation_seeds,
    vehicle_counts,
    training_strategies,
    evaluation_strategies,
    parallelism: int,
    *,
    dry_run: bool = False,
) -> None:
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.write_text(
        _render_pipeline(
            profile_path,
            output,
            study,
            training_seeds,
            evaluation_seeds,
            vehicle_counts,
            training_strategies,
            evaluation_strategies,
            parallelism,
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(repository / "scripts" / "run-final-multiseed.py"),
        "--config",
        str(pipeline_path),
        "--base-config",
        str(profile_path),
        "--parallelism",
        str(parallelism),
    ]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=repository)
    if completed.returncode:
        raise RuntimeError(
            f"multi-seed stage failed with code {completed.returncode}: {pipeline_path}"
        )


def _render_pipeline(
    profile: Path,
    output: Path,
    study: dict,
    training_seeds,
    evaluation_seeds,
    vehicles,
    training_strategies,
    evaluation_strategies,
    parallelism: int,
) -> str:
    return "\n".join(
        [
            "[pipeline]",
            f"base_config = {json.dumps(profile.as_posix())}",
            f"output_dir = {json.dumps(output.as_posix())}",
            f"training_steps = {int(study['steps'])}",
            f"evaluation_steps = {int(study['steps'])}",
            f"vehicle_counts = {json.dumps(list(vehicles))}",
            f"training_seeds = {json.dumps(list(training_seeds))}",
            f"evaluation_seeds = {json.dumps(list(evaluation_seeds))}",
            f"training_strategies = {json.dumps(list(training_strategies))}",
            f"evaluation_strategies = {json.dumps(list(evaluation_strategies))}",
            f"training_task_sample_rate = {float(study['training_task_sample_rate'])}",
            f"evaluation_task_sample_rate = {float(study['evaluation_task_sample_rate'])}",
            f"parallelism = {parallelism}",
            f"minimum_free_disk_gb = {float(study['minimum_free_disk_gb'])}",
            f"storage_upper_bound_gb = {float(study['storage_upper_bound_gb'])}",
            "allow_strategy_subset = true",
            "",
        ]
    )


def _hybrid_screen_metrics(output: Path) -> dict[str, float]:
    rows = _read_csv(output / "aggregate-results.csv")
    row = next(row for row in rows if row["strategy"] == "hybrid_stackelberg")
    success = float(row["success_rate_mean"])
    std = float(row["success_rate_sample_std"])
    diagnostics = _diagnostic_rollup(output)
    return {
        "success": success,
        "std": std,
        "robust_success": success - std,
        "oracle": float(row["oracle_success_rate_mean"]),
        "queue": float(row["avg_cloud_queue_length_mean"]),
        "reward": float(row["avg_reward_mean"]),
        "v2i": float(row["v2i_offload_ratio_mean"]),
        "deviation": float(row["hybrid_deviation_ratio_mean"]),
        **diagnostics,
    }


def _diagnostic_rollup(output: Path) -> dict[str, float]:
    evaluation_rows = [
        row
        for row in _read_csv(output / "evaluation-results.csv")
        if row["strategy"] == "hybrid_stackelberg"
    ]
    agreement = [0, 0]
    follows_game = [0, 0]
    follows_dqn = [0, 0]
    early_success = []
    late_success = []
    q_margins = []
    pricing_mae = []
    response_residuals = []
    outer_residuals = []
    for row in evaluation_rows:
        run_dir = Path(row["run_dir"])
        decision_path = run_dir / "decision-diagnostics.json"
        pricing_path = run_dir / "pricing-diagnostics.json"
        if decision_path.is_file():
            diagnostic = json.loads(decision_path.read_text(encoding="utf-8"))
            matrices = diagnostic["global_action_matrices"]
            _matrix_agreement(matrices["game_to_dqn"], agreement)
            _matrix_agreement(matrices["game_to_final"], follows_game)
            _matrix_agreement(matrices["dqn_to_final"], follows_dqn)
            windows = diagnostic.get("windows", [])
            if windows:
                early_success.append(float(windows[0]["success_rate"]))
                late_success.append(float(windows[-1]["success_rate"]))
                q_margins.extend(
                    float(window["avg_dqn_q_margin"])
                    for window in windows
                    if float(window["dqn_decision_ratio"]) > 0.0
                )
        if pricing_path.is_file():
            pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
            for window in pricing.get("windows", []):
                pricing_mae.append(float(window["weighted_prediction_mae"]))
                response_residuals.append(
                    abs(float(window["avg_response_cycle_residual"]))
                )
                outer_residuals.append(abs(float(window["avg_outer_cycle_residual"])))
    return {
        "game_dqn_agreement": _ratio(*agreement),
        "final_follows_game": _ratio(*follows_game),
        "final_follows_dqn": _ratio(*follows_dqn),
        "early_success": _mean(early_success),
        "late_success": _mean(late_success),
        "avg_q_margin": _mean(q_margins),
        "pricing_prediction_mae": _mean(pricing_mae),
        "pricing_response_residual": _mean(response_residuals),
        "pricing_outer_residual": _mean(outer_residuals),
    }


def _matrix_agreement(matrix: dict[str, dict[str, int]], result: list[int]) -> None:
    for source, targets in matrix.items():
        for target, count in targets.items():
            value = int(count)
            result[1] += value
            result[0] += value if source == target else 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _select_screen_winner(
    results: dict[str, dict[str, float]],
    min_material_gain: float = 0.0,
) -> str:
    ranked = _rank_screen_profiles(results)
    best_score = results[ranked[0]]["robust_success"]
    # Dict order encodes increasing model complexity. Prefer the simplest
    # candidate statistically indistinguishable from the best screen score.
    return next(
        name
        for name in results
        if results[name]["robust_success"] >= best_score - min_material_gain
    )


def _rank_screen_profiles(results: dict[str, dict[str, float]]) -> list[str]:
    return sorted(
        results,
        key=lambda name: (
            results[name]["robust_success"],
            results[name]["success"],
            results[name]["reward"],
            -results[name]["queue"],
        ),
        reverse=True,
    )


def _confirmation_metrics(output: Path) -> dict[int, dict[str, dict[str, float]]]:
    rows = _read_csv(output / "aggregate-results.csv")
    result: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        vehicles = int(row["vehicle_count"])
        result.setdefault(vehicles, {})[row["strategy"]] = {
            "success": float(row["success_rate_mean"]),
            "std": float(row["success_rate_sample_std"]),
            "reward": float(row["avg_reward_mean"]),
            "queue": float(row["avg_cloud_queue_length_mean"]),
            "oracle": float(row["oracle_success_rate_mean"]),
        }
    return result


def _select_confirmation_winner(
    results: dict[str, dict],
    min_material_gain: float = 0.0,
) -> str:
    def score(name: str):
        loads = results[name]
        deltas = []
        hybrid_success = []
        rewards = []
        for strategies in loads.values():
            hybrid = strategies["hybrid_stackelberg"]
            comparator = max(
                strategies["dqn"]["success"],
                strategies["stackelberg"]["success"],
            )
            deltas.append(hybrid["success"] - comparator)
            hybrid_success.append(hybrid["success"])
            rewards.append(hybrid["reward"])
        return (
            min(deltas),
            sum(hybrid_success) / len(hybrid_success),
            sum(rewards) / len(rewards),
        )

    scored = {name: score(name) for name in results}
    best_worst_delta = max(value[0] for value in scored.values())
    return next(
        name
        for name in results
        if scored[name][0] >= best_worst_delta - min_material_gain
    )


def _serverless_gate(
    final_metrics: dict,
    confirmation: dict,
    selected_name: str,
    success_tolerance: float,
    current_regression_tolerance: float,
) -> tuple[bool, dict]:
    load_deltas = {}
    for vehicles, strategies in final_metrics.items():
        if vehicles < 2000:
            continue
        hybrid = strategies["hybrid_stackelberg"]["success"]
        comparator = max(strategies["dqn"]["success"], strategies["stackelberg"]["success"])
        load_deltas[str(vehicles)] = hybrid - comparator
    current_regressions = {}
    if "current" in confirmation and selected_name in confirmation:
        for vehicles in confirmation[selected_name]:
            selected = confirmation[selected_name][vehicles]["hybrid_stackelberg"]["success"]
            current = confirmation["current"][vehicles]["hybrid_stackelberg"]["success"]
            current_regressions[str(vehicles)] = selected - current
    within_baseline_tolerance = all(
        value >= -success_tolerance for value in load_deltas.values()
    )
    no_current_regression = all(
        value >= -current_regression_tolerance for value in current_regressions.values()
    )
    gate = {
        "success_tolerance": success_tolerance,
        "current_regression_tolerance": current_regression_tolerance,
        "hybrid_minus_best_baseline": load_deltas,
        "selected_minus_current_hybrid": current_regressions,
        "within_baseline_tolerance": within_baseline_tolerance,
        "no_current_regression": no_current_regression,
        "claim_strictly_strongest": bool(load_deltas) and all(value > 0 for value in load_deltas.values()),
    }
    return within_baseline_tolerance and no_current_regression, gate


def _unique_profiles(profiles: dict[str, Path]) -> dict[str, Path]:
    seen = set()
    result = {}
    for name, path in profiles.items():
        signature = json.dumps(
            SimulationConfig.from_toml(path).to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        result[name] = path
    return result


def _write_selection(path: Path, results: dict, winner: str) -> None:
    rows = []
    for name, metrics in results.items():
        rows.append({"profile": name, **metrics, "selected": int(name == winner)})
    _write_csv(path, rows)


def _write_confirmation_selection(path: Path, results: dict, winner: str) -> None:
    rows = []
    for name, loads in results.items():
        for vehicles, strategies in loads.items():
            hybrid = strategies["hybrid_stackelberg"]
            dqn = strategies["dqn"]
            stack = strategies["stackelberg"]
            rows.append(
                {
                    "profile": name,
                    "vehicle_count": vehicles,
                    "hybrid_success": hybrid["success"],
                    "dqn_success": dqn["success"],
                    "stackelberg_success": stack["success"],
                    "hybrid_minus_dqn": hybrid["success"] - dqn["success"],
                    "hybrid_minus_stackelberg": hybrid["success"] - stack["success"],
                    "selected": int(name == winner),
                }
            )
    _write_csv(path, rows)


def _write_study_summary(
    path: Path,
    causal: dict,
    causal_winner: str,
    rl: dict,
    rl_winner: str,
    confirmation: dict,
    final_profile: str,
    final_metrics: dict,
    gate: dict,
    serverless_ready: bool,
) -> None:
    lines = [
        "# Staged Hybrid optimization study",
        "",
        f"- Causal winner: `{causal_winner}`",
        f"- RL winner: `{rl_winner}`",
        f"- Final profile: `{final_profile}`",
        f"- Serverless continuation: `{serverless_ready}`",
        "",
        "## Causal screen",
        "",
        "| Profile | Success | Std | Robust | Oracle | Queue | Reward | V2I | G=D | Final=G | Final=DQN | Late | Price MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(causal.items()):
        lines.append(
            f"| {name} | {100*row['success']:.2f}% | {100*row['std']:.2f} pp | "
            f"{100*row['robust_success']:.2f}% | {100*row['oracle']:.2f}% | "
            f"{row['queue']:.2f} | {row['reward']:.3f} | {100*row['v2i']:.2f}% | "
            f"{100*row['game_dqn_agreement']:.2f}% | "
            f"{100*row['final_follows_game']:.2f}% | "
            f"{100*row['final_follows_dqn']:.2f}% | "
            f"{100*row['late_success']:.2f}% | "
            f"{row['pricing_prediction_mae']:.3f} |"
        )
    lines.extend([
        "",
        "## RL screen",
        "",
        "| Profile | Success | Std | Robust | Queue | Reward | G=D | Final=DQN | Late | Q margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in sorted(rl.items()):
        lines.append(
            f"| {name} | {100*row['success']:.2f}% | {100*row['std']:.2f} pp | "
            f"{100*row['robust_success']:.2f}% | {row['queue']:.2f} | "
            f"{row['reward']:.3f} | {100*row['game_dqn_agreement']:.2f}% | "
            f"{100*row['final_follows_dqn']:.2f}% | "
            f"{100*row['late_success']:.2f}% | {row['avg_q_margin']:.3f} |"
        )
    lines.extend(["", "## Confirmation", "", "| Profile | Vehicles | Hybrid | DQN | Stackelberg | H-DQN | H-Stack |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, loads in sorted(confirmation.items()):
        for vehicles, strategies in sorted(loads.items()):
            h = strategies["hybrid_stackelberg"]["success"]
            d = strategies["dqn"]["success"]
            s = strategies["stackelberg"]["success"]
            lines.append(
                f"| {name} | {vehicles} | {100*h:.2f}% | {100*d:.2f}% | "
                f"{100*s:.2f}% | {100*(h-d):+.2f} pp | {100*(h-s):+.2f} pp |"
            )
    lines.extend(["", "## Final independent matrix", "", "| Vehicles | Strategy | Success |", "|---:|---|---:|"])
    for vehicles, strategies in sorted(final_metrics.items()):
        for strategy, metrics in strategies.items():
            lines.append(f"| {vehicles} | {strategy} | {100*metrics['success']:.2f}% |")
    lines.extend(["", "## Serverless gate", "", "```json", json.dumps(gate, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate.resolve()
    raise RuntimeError(f"could not locate repository root from {path}")


def _resolve(base: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _check_free_disk(path: Path, minimum_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB is free; minimum_free_disk_gb is {minimum_gb:.2f}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
