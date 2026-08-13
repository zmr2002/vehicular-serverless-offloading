"""Summarize the cross-checkpoint arm evaluations against the final matrix.

Reads every completed run under results/verified/hybrid-cross-checkpoint-eval
plus the original per-replicate baselines from
results/verified/hybrid-optimization-study/final/evaluation-results.csv and
writes arms-summary.md/csv next to the arm runs.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
from statistics import fmean, stdev

REPO = Path(__file__).resolve().parents[1]
ARMS_ROOT = REPO / "results" / "verified" / "hybrid-cross-checkpoint-eval"
FINAL_RESULTS = (
    REPO
    / "results"
    / "verified"
    / "hybrid-optimization-study"
    / "final"
    / "evaluation-results.csv"
)
CASE_PATTERN = re.compile(
    r"^v(?P<vehicles>\d+)-rep(?P<replicate>\d)-(?P<arm>[a-z-]+)-seed(?P<seed>\d+)$"
)
BASELINE_STRATEGIES = ("dqn", "stackelberg", "hybrid_stackelberg")


def load_arm_rows() -> list[dict]:
    rows = []
    for summary_path in sorted(ARMS_ROOT.glob("*/*/summary.json")):
        case = summary_path.parent.parent.name
        match = CASE_PATTERN.match(case)
        if not match:
            continue
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "case": case,
                "arm": match["arm"],
                "vehicles": int(match["vehicles"]),
                "replicate": int(match["replicate"]),
                "evaluation_seed": int(match["seed"]),
                "success_rate": float(summary["success_rate"]),
                "avg_success_latency_s": float(summary["avg_success_latency_s"]),
                "avg_energy_j": float(summary["avg_energy_j"]),
                "avg_cost_per_task": float(summary["avg_cost_per_task"]),
                "avg_reward": float(summary["avg_reward"]),
                "dqn_decision_ratio": float(summary.get("dqn_decision_ratio", 0.0)),
            }
        )
    return rows


def load_baselines() -> dict[tuple[int, int, str], float]:
    baselines: dict[tuple[int, int, str], float] = {}
    with FINAL_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["strategy"] not in BASELINE_STRATEGIES:
                continue
            key = (
                int(row["configured_vehicle_count"]),
                int(row["replicate"]),
                row["strategy"],
            )
            baselines[key] = float(row["success_rate"])
    return baselines


def main() -> int:
    rows = load_arm_rows()
    if not rows:
        print(f"no completed arm runs under {ARMS_ROOT}")
        return 1
    baselines = load_baselines()
    for row in rows:
        vehicles, replicate = row["vehicles"], row["replicate"]
        best_baseline = max(
            (
                baselines.get((vehicles, replicate, strategy), float("nan")),
                strategy,
            )
            for strategy in ("dqn", "stackelberg")
        )
        row["original_hybrid"] = baselines.get(
            (vehicles, replicate, "hybrid_stackelberg")
        )
        row["best_baseline"] = best_baseline[0]
        row["best_baseline_strategy"] = best_baseline[1]
        row["margin_over_best_baseline"] = (
            row["success_rate"] - best_baseline[0]
        )

    fields = list(rows[0].keys())
    with (ARMS_ROOT / "arms-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["arm"], r["vehicles"], r["replicate"])))

    lines = [
        "# Cross-checkpoint arm summary",
        "",
        "Baselines come from the final matrix replicates "
        "(results/verified/hybrid-optimization-study/final).",
        "",
        "| Arm | Vehicles | Rep | Success | Original hybrid | Best baseline | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["arm"], r["vehicles"], r["replicate"])):
        lines.append(
            f"| {row['arm']} | {row['vehicles']} | {row['replicate']} | "
            f"{100 * row['success_rate']:.2f}% | "
            f"{100 * row['original_hybrid']:.2f}% | "
            f"{100 * row['best_baseline']:.2f}% "
            f"({row['best_baseline_strategy']}) | "
            f"{100 * row['margin_over_best_baseline']:+.2f} pp |"
        )
    lines.extend(["", "## Per-arm aggregate margin over the best baseline", ""])
    lines.append("| Arm | Vehicles | Mean success | Std | Mean margin |")
    lines.append("|---|---:|---:|---:|---:|")
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["vehicles"]), []).append(row)
    for (arm, vehicles), group in sorted(grouped.items()):
        successes = [row["success_rate"] for row in group]
        margins = [row["margin_over_best_baseline"] for row in group]
        spread = stdev(successes) if len(successes) > 1 else 0.0
        lines.append(
            f"| {arm} | {vehicles} | {100 * fmean(successes):.2f}% | "
            f"{100 * spread:.2f} pp | {100 * fmean(margins):+.2f} pp |"
        )
    lines.append("")
    (ARMS_ROOT / "arms-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SUMMARY {ARMS_ROOT / 'arms-summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
