"""A/B screen of the per-step exploration schedule for DQN training.

Retrains the pure DQN at 2,000 vehicles on the SAME training seeds as the
fresh matrix (31641-43) with `epsilon_decay_per_step` enabled, evaluates each
checkpoint on its paired evaluation seed (84-86), and compares against the
already completed base-recipe DQN evaluations in
results/verified/final-decoupled. Pre-declared rule (matching the
hybrid-optimization-study precedent): the per-step recipe is selected only if
its mean success exceeds the base recipe by more than 0.25 pp; otherwise the
simpler base recipe is kept. Prints the winning base-config path as the last
stdout line.

Cases resume by existing summary.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO / "results" / "verified" / "eps-screen"
EPS_PROFILE = REPO / "configs" / "hybrid-decoupled-eps.toml"
TRAIN_PROFILE = REPO / "configs" / "eps-screen-train.toml"
BASE_PROFILE = REPO / "configs" / "hybrid-decoupled.toml"
EVAL_PROFILE = REPO / "configs" / "cross-checkpoint-eval-adequacy.toml"
FRESH = REPO / "results" / "verified" / "final-decoupled"
PAIRS = ((1, 31641, 84), (2, 31642, 85), (3, 31643, 86))
VEHICLES = 2000
STEPS = 2000
SELECTION_MARGIN = 0.0025


def _run(command: list[str], log_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")
    with open(log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command, cwd=str(REPO), env=environment, stdout=log_file, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}; see {log_path}")


def _summary(output_dir: Path) -> dict | None:
    matches = glob.glob(str(output_dir / "*" / "summary.json"))
    if not matches:
        return None
    with open(matches[0], encoding="utf-8") as handle:
        return json.load(handle)


def _simulate_args(mode: str, seed: int, extra: list[str], output_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "vehicular_offloading", "simulate",
        "--config", str(EVAL_PROFILE if mode == "evaluate" else TRAIN_PROFILE),
        "--strategy", "dqn", "--mobility", "sumo",
        "--steps", str(STEPS), "--vehicles", str(VEHICLES),
        "--seed", str(seed), "--dqn-mode", mode,
        "--output-dir", str(output_dir), *extra,
    ]


def screen_case(replicate: int, training_seed: int, evaluation_seed: int) -> float:
    train_dir = OUTPUT_ROOT / f"train-eps-rep{replicate}-seed{training_seed}"
    if _summary(train_dir) is None:
        train_dir.mkdir(parents=True, exist_ok=True)
        print(f"TRAIN eps rep{replicate} seed={training_seed}", flush=True)
        _run(_simulate_args("train", training_seed, [], train_dir), train_dir / "driver.log")
    checkpoints = glob.glob(str(train_dir / "*" / "dqn-policy.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one checkpoint under {train_dir}, found {checkpoints}")
    eval_dir = OUTPUT_ROOT / f"eval-eps-rep{replicate}-seed{evaluation_seed}"
    summary = _summary(eval_dir)
    if summary is None:
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"EVAL eps rep{replicate} seed={evaluation_seed}", flush=True)
        _run(
            _simulate_args("evaluate", evaluation_seed, ["--checkpoint", checkpoints[0]], eval_dir),
            eval_dir / "driver.log",
        )
        summary = _summary(eval_dir)
    assert summary is not None
    return float(summary["success_rate"])


def base_success(replicate: int) -> float:
    pattern = str(
        FRESH
        / f"replicate-{replicate:02d}"
        / "run-*"
        / "evaluation"
        / f"dqn-{VEHICLES}"
        / "*"
        / "summary.json"
    )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError(f"expected one base DQN evaluation for {pattern}, found {matches}")
    with open(matches[0], encoding="utf-8") as handle:
        return float(json.load(handle)["success_rate"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        for replicate, training_seed, evaluation_seed in PAIRS:
            print(
                f"[plan] rep{replicate} train seed {training_seed} (eps recipe) "
                f"vs base DQN, paired eval seed {evaluation_seed}; "
                f"base success {100 * base_success(replicate):.2f}%"
            )
        print(str(BASE_PROFILE))
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        eps_values = list(pool.map(lambda pair: screen_case(*pair), PAIRS))
    base_values = [base_success(replicate) for replicate, _, _ in PAIRS]
    for (replicate, _, _), eps, base in zip(PAIRS, eps_values, base_values):
        print(f"rep{replicate}: eps {100 * eps:.2f}% vs base {100 * base:.2f}%")
    eps_mean, base_mean = fmean(eps_values), fmean(base_values)
    print(
        f"means: eps {100 * eps_mean:.2f}% vs base {100 * base_mean:.2f}% "
        f"(selection threshold +{100 * SELECTION_MARGIN:.2f} pp)"
    )
    winner = EPS_PROFILE if eps_mean > base_mean + SELECTION_MARGIN else BASE_PROFILE
    print(f"selected {'per-step epsilon' if winner == EPS_PROFILE else 'base'} recipe")
    print(str(winner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
