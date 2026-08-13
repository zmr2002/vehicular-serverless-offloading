"""Evaluate Hybrid arbitration variants against existing trained checkpoints.

Sources:
- `study` (default): the hybrid-optimization-study final matrix (evaluation
  seeds 81-83). Both a pure-DQN and a co-trained Hybrid checkpoint exist per
  replicate, so the coupling, the refutation defense, and the adequacy
  arbitration can be separated without training anything.
- `fresh`: the fresh-seed decoupled matrix results/verified/final-decoupled
  (evaluation seeds 84-86). Only pure-DQN checkpoints exist; hybrid rows can
  be re-evaluated under a corrected arbitration and merged with the already
  completed baseline evaluations by summarize_fresh_final.py.

Cases resume by existing summary.json. Run with the pipeline venv:
    <venv>/python.exe scripts/run_cross_checkpoint_eval.py
        [--arm dqnckpt|dqnckpt-reliability|internal-reliability|
              dqnckpt-adequacy|dqnckpt-damping]
        [--source study|fresh] [--workers 3] [--dry-run]
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

REPO = Path(__file__).resolve().parents[1]

ARMS = {
    "dqnckpt": {
        "config": "cross-checkpoint-eval.toml",
        "checkpoint_strategy": "dqn",
        "suffix": "dqnckpt",
    },
    "dqnckpt-reliability": {
        "config": "cross-checkpoint-eval-reliability.toml",
        "checkpoint_strategy": "dqn",
        "suffix": "dqnckpt-rel",
    },
    "internal-reliability": {
        "config": "cross-checkpoint-eval-reliability.toml",
        "checkpoint_strategy": "hybrid_stackelberg",
        "suffix": "hybridckpt-rel",
    },
    "dqnckpt-adequacy": {
        "config": "cross-checkpoint-eval-adequacy.toml",
        "checkpoint_strategy": "dqn",
        "suffix": "dqnckpt-adq",
    },
    "dqnckpt-damping": {
        "config": "cross-checkpoint-eval-adequacy-damping.toml",
        "checkpoint_strategy": "dqn",
        "suffix": "dqnckpt-damp",
    },
    "dqnckpt-cap": {
        "config": "cross-checkpoint-eval-cap.toml",
        "checkpoint_strategy": "dqn",
        "suffix": "dqnckpt-cap",
    },
}

SOURCES = {
    "study": {
        "checkpoint_root": REPO
        / "results"
        / "verified"
        / "hybrid-optimization-study"
        / "final",
        "output_root": REPO
        / "results"
        / "verified"
        / "hybrid-cross-checkpoint-eval",
        "evaluation_seeds": {1: 81, 2: 82, 3: 83},
    },
    "fresh": {
        "checkpoint_root": REPO / "results" / "verified" / "final-decoupled",
        "output_root": REPO / "results" / "verified" / "hybrid-fresh-reeval",
        "evaluation_seeds": {1: 84, 2: 85, 3: 86},
    },
}

VEHICLE_COUNTS = (1000, 2000, 4000)
STEPS = 2000


def training_checkpoint(source: str, replicate: int, vehicles: int, strategy: str) -> Path:
    pattern = str(
        SOURCES[source]["checkpoint_root"]
        / f"replicate-{replicate:02d}"
        / "run-*"
        / "training"
        / f"{strategy}-{vehicles}"
        / "*"
        / "dqn-policy.pt"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one checkpoint for {pattern}, found {matches}")
    return Path(matches[0])


def case_output_dir(source: str, arm: str, replicate: int, vehicles: int, seed: int) -> Path:
    suffix = ARMS[arm]["suffix"]
    return (
        SOURCES[source]["output_root"]
        / f"v{vehicles}-rep{replicate}-{suffix}-seed{seed}"
    )


def is_complete(output_dir: Path) -> bool:
    return bool(glob.glob(str(output_dir / "*" / "summary.json")))


def run_case(source: str, arm: str, replicate: int, vehicles: int) -> tuple[str, float | None]:
    seed = SOURCES[source]["evaluation_seeds"][replicate]
    output_dir = case_output_dir(source, arm, replicate, vehicles, seed)
    label = output_dir.name
    if is_complete(output_dir):
        return label, _read_success(output_dir)
    checkpoint = training_checkpoint(
        source, replicate, vehicles, ARMS[arm]["checkpoint_strategy"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "vehicular_offloading",
        "simulate",
        "--config",
        str(REPO / "configs" / ARMS[arm]["config"]),
        "--strategy",
        "hybrid_stackelberg",
        "--mobility",
        "sumo",
        "--steps",
        str(STEPS),
        "--vehicles",
        str(vehicles),
        "--seed",
        str(seed),
        "--dqn-mode",
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")
    log_path = output_dir / "driver.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command, cwd=str(REPO), env=environment, stdout=log_file, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {log_path}")
    return label, _read_success(output_dir)


def _read_success(output_dir: Path) -> float | None:
    summaries = glob.glob(str(output_dir / "*" / "summary.json"))
    if not summaries:
        return None
    with open(summaries[0], encoding="utf-8") as handle:
        return json.load(handle).get("success_rate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="dqnckpt")
    parser.add_argument("--source", choices=sorted(SOURCES), default="study")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.source == "fresh" and ARMS[args.arm]["checkpoint_strategy"] != "dqn":
        raise SystemExit("the fresh source only provides pure-DQN checkpoints")

    replicates = sorted(SOURCES[args.source]["evaluation_seeds"])
    cases = [(replicate, vehicles) for vehicles in VEHICLE_COUNTS for replicate in replicates]
    for replicate, vehicles in cases:
        checkpoint = training_checkpoint(
            args.source, replicate, vehicles, ARMS[args.arm]["checkpoint_strategy"]
        )
        seed = SOURCES[args.source]["evaluation_seeds"][replicate]
        state = (
            "done"
            if is_complete(
                case_output_dir(args.source, args.arm, replicate, vehicles, seed)
            )
            else "pending"
        )
        print(
            f"[{state}] source={args.source} arm={args.arm} v{vehicles} "
            f"rep{replicate} seed{seed} checkpoint={checkpoint}"
        )
    if args.dry_run:
        return 0

    results: dict[str, float | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_case, args.source, args.arm, replicate, vehicles): (
                replicate,
                vehicles,
            )
            for replicate, vehicles in cases
        }
        for future in concurrent.futures.as_completed(futures):
            label, success = future.result()
            results[label] = success
            print(f"DONE {label} success={success}")

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
