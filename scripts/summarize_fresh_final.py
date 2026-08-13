"""Merge re-evaluated Hybrid rows into the fresh-seed final matrix.

Baseline evaluations (random/greedy/dqn/stackelberg) come from
results/verified/final-decoupled/evaluation-results.csv unchanged. The
hybrid_stackelberg rows are replaced by the corrected-arbitration re-runs
under results/verified/hybrid-fresh-reeval (same checkpoints, mobility, task
streams, and evaluation seeds). Writes corrected-final-summary.md and
corrected-evaluation-results.csv next to the original summary.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "results" / "verified" / "final-decoupled"
REEVAL = REPO / "results" / "verified" / "hybrid-fresh-reeval"
STRATEGIES = ("random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg")
CASE_PATTERN = re.compile(
    r"^v(?P<vehicles>\d+)-rep(?P<replicate>\d)-(?P<arm>[a-z-]+)-seed(?P<seed>\d+)$"
)
METRICS = (
    "success_rate",
    "avg_success_latency_s",
    "avg_energy_j",
    "avg_cost_per_task",
    "avg_reward",
)
T_CRITICAL_95 = {2: 12.706205, 3: 4.302653, 4: 3.182446, 5: 2.776445}


def load_rows() -> list[dict]:
    with (FINAL / "evaluation-results.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    replaced = 0
    reeval: dict[tuple[int, int], dict] = {}
    arms = set()
    for summary_path in sorted(REEVAL.glob("*/*/summary.json")):
        match = CASE_PATTERN.match(summary_path.parent.parent.name)
        if not match:
            continue
        arms.add(match["arm"])
        with summary_path.open(encoding="utf-8") as handle:
            reeval[(int(match["vehicles"]), int(match["replicate"]))] = json.load(handle)
    if len(arms) > 1:
        raise SystemExit(f"multiple re-evaluation arms present, refusing to mix: {sorted(arms)}")
    if not reeval:
        raise SystemExit(f"no completed re-evaluation runs under {REEVAL}")
    for row in rows:
        if row["strategy"] != "hybrid_stackelberg":
            continue
        key = (int(row["configured_vehicle_count"]), int(row["replicate"]))
        summary = reeval.get(key)
        if summary is None:
            raise SystemExit(f"missing re-evaluation for vehicles={key[0]} replicate={key[1]}")
        if int(summary["total_tasks"]) != int(row["total_tasks"]):
            raise SystemExit(f"task stream mismatch for vehicles={key[0]} replicate={key[1]}")
        for metric in set(METRICS) | {
            "local_offload_ratio", "v2v_offload_ratio", "v2i_offload_ratio",
            "oracle_success_rate", "avoidable_failure_rate", "dqn_decision_ratio",
            "hybrid_deviation_ratio", "hybrid_beneficial_deviation_rate",
        }:
            if metric in summary and metric in row:
                row[metric] = summary[metric]
        replaced += 1
    if replaced != len(reeval):
        raise SystemExit(f"replaced {replaced} rows but found {len(reeval)} re-evaluations")
    print(f"replaced {replaced} hybrid rows from arm {next(iter(arms))}")
    return rows


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    mean_value = statistics.fmean(values)
    if len(values) == 1:
        return mean_value, 0.0, 0.0
    sample_std = statistics.stdev(values)
    critical = T_CRITICAL_95.get(len(values), 1.959964)
    return mean_value, sample_std, critical * sample_std / math.sqrt(len(values))


def main() -> int:
    rows = load_rows()
    with (FINAL / "corrected-evaluation-results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    vehicles = sorted({int(row["configured_vehicle_count"]) for row in rows})
    replicates = sorted({int(row["replicate"]) for row in rows})
    indexed = {
        (int(row["replicate"]), int(row["configured_vehicle_count"]), row["strategy"]): row
        for row in rows
    }
    lines = [
        "# Corrected fresh-seed final matrix (decoupled Hybrid re-evaluation)",
        "",
        "Baselines are unchanged from `final-summary.md`; hybrid_stackelberg is",
        "re-evaluated with the corrected arbitration on the same checkpoints,",
        "mobility, task streams, and evaluation seeds.",
        "",
        "## Aggregate success",
        "",
        "| Vehicles | Strategy | Success (95% CI) | Energy (J) | Cost/task |",
        "|---:|---|---:|---:|---:|",
    ]
    for count in vehicles:
        for strategy in STRATEGIES:
            values = [
                float(indexed[(replicate, count, strategy)]["success_rate"])
                for replicate in replicates
            ]
            energy = statistics.fmean(
                float(indexed[(replicate, count, strategy)]["avg_energy_j"])
                for replicate in replicates
            )
            cost = statistics.fmean(
                float(indexed[(replicate, count, strategy)]["avg_cost_per_task"])
                for replicate in replicates
            )
            mean_value, _, ci95 = mean_std_ci(values)
            lines.append(
                f"| {count} | {strategy} | {100 * mean_value:.2f}% ± "
                f"{100 * ci95:.2f} pp | {energy:.2f} | {cost:.4f} |"
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
    for count in vehicles:
        for comparator in STRATEGIES[:-1]:
            differences = [
                float(indexed[(replicate, count, "hybrid_stackelberg")]["success_rate"])
                - float(indexed[(replicate, count, comparator)]["success_rate"])
                for replicate in replicates
            ]
            mean_value, _, ci95 = mean_std_ci(differences)
            lines.append(
                f"| {count} | {comparator} | {100 * mean_value:+.2f} ± "
                f"{100 * ci95:.2f} pp |"
            )
    lines.append("")
    (FINAL / "corrected-final-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SUMMARY {FINAL / 'corrected-final-summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
