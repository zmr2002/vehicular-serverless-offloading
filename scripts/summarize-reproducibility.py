from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev


T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate completed training/evaluation replications"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-replications", type=int, default=3)
    args = parser.parse_args()

    sessions = _find_sessions(args.input)
    if len(sessions) != args.expected_replications:
        raise RuntimeError(
            f"expected {args.expected_replications} complete sessions, "
            f"found {len(sessions)}: {sessions}"
        )

    replicate_rows: list[dict] = []
    for session in sessions:
        training_seed = _single_seed(session / "training-results.csv")
        with (session / "evaluation-results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                replicate_rows.append(
                    {
                        "session": str(session),
                        "training_seed": training_seed,
                        "evaluation_seed": int(row["seed"]),
                        "strategy": row["strategy"],
                        "vehicle_count": int(row["configured_vehicle_count"]),
                        "success_rate": float(row["success_rate"]),
                        "avg_latency_s": float(row["avg_latency_s"]),
                        "avg_energy_j": float(row["avg_energy_j"]),
                        "avg_cost_per_task": float(row["avg_cost_per_task"]),
                        "avg_reward": float(row["avg_reward"]),
                    }
                )

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in replicate_rows:
        grouped[(row["strategy"], row["vehicle_count"])].append(row)

    summary_rows: list[dict] = []
    for (strategy, vehicles), rows in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if len(rows) != args.expected_replications:
            raise RuntimeError(
                f"{strategy}/{vehicles} has {len(rows)} replications, "
                f"expected {args.expected_replications}"
            )
        values = [row["success_rate"] for row in rows]
        sample_std = stdev(values)
        critical = T_CRITICAL_95.get(len(values), 1.96)
        ci_half_width = critical * sample_std / sqrt(len(values))
        summary_rows.append(
            {
                "strategy": strategy,
                "vehicle_count": vehicles,
                "replications": len(rows),
                "evaluation_seeds": ",".join(
                    str(row["evaluation_seed"]) for row in rows
                ),
                "mean_success_rate": mean(values),
                "sample_std": sample_std,
                "ci95_half_width": ci_half_width,
                "min_success_rate": min(values),
                "max_success_rate": max(values),
                "mean_latency_s": mean(row["avg_latency_s"] for row in rows),
                "mean_energy_j": mean(row["avg_energy_j"] for row in rows),
                "mean_cost_per_task": mean(
                    row["avg_cost_per_task"] for row in rows
                ),
                "mean_reward": mean(row["avg_reward"] for row in rows),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "replicate-results.csv", replicate_rows)
    _write_csv(args.output_dir / "reproducibility-summary.csv", summary_rows)
    (args.output_dir / "reproducibility-summary.md").write_text(
        _markdown(replicate_rows, summary_rows), encoding="utf-8"
    )
    print(args.output_dir / "reproducibility-summary.md")
    return 0


def _find_sessions(inputs: list[Path]) -> list[Path]:
    sessions: set[Path] = set()
    for value in inputs:
        root = value.resolve()
        candidates = (
            [root / "evaluation-results.csv"]
            if root.is_dir()
            else []
        )
        if root.is_dir():
            candidates.extend(root.rglob("evaluation-results.csv"))
        for result_path in candidates:
            session = result_path.parent
            required = [
                session / "training-results.csv",
                session / "diagnostic-summary.json",
            ]
            if all(path.exists() for path in required):
                sessions.add(session)
    return sorted(sessions)


def _single_seed(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        seeds = {int(row["seed"]) for row in csv.DictReader(handle)}
    if len(seeds) != 1:
        raise RuntimeError(f"expected one training seed in {path}, found {seeds}")
    return seeds.pop()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(replicates: list[dict], summaries: list[dict]) -> str:
    lines = [
        "# Adaptive-gate reproducibility",
        "",
        "Each replication uses an independently trained policy and an independent "
        "evaluation task stream. Strategies within one replication share the "
        "same evaluation seed.",
        "",
        "## Replications",
        "",
        "| Training seed | Evaluation seed | Session |",
        "|---:|---:|---|",
    ]
    seen = set()
    for row in replicates:
        key = (row["training_seed"], row["evaluation_seed"], row["session"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {key[0]} | {key[1]} | `{key[2]}` |")

    for vehicles in sorted({row["vehicle_count"] for row in summaries}):
        lines.extend(
            [
                "",
                f"## {vehicles} vehicles",
                "",
                "| Strategy | Mean success | 95% CI | Std. dev. | Range |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        vehicle_rows = [
            row for row in summaries if row["vehicle_count"] == vehicles
        ]
        vehicle_rows.sort(
            key=lambda row: row["mean_success_rate"], reverse=True
        )
        for row in vehicle_rows:
            lines.append(
                f"| {row['strategy']} | "
                f"{100 * row['mean_success_rate']:.2f}% | "
                f"±{100 * row['ci95_half_width']:.2f} pp | "
                f"{100 * row['sample_std']:.2f} pp | "
                f"{100 * row['min_success_rate']:.2f}–"
                f"{100 * row['max_success_rate']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "With only three replications, the Student-t confidence intervals are "
            "intentionally wide. Use the per-seed values and ordering together "
            "with the interval rather than treating the mean as conclusive.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
