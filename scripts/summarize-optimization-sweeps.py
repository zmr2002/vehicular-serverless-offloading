from __future__ import annotations

import argparse
import csv
from pathlib import Path


PAPER_SUCCESS = {
    ("hybrid_stackelberg", 1000): 0.8567,
    ("stackelberg", 1000): 0.8413,
    ("dqn", 1000): 0.8090,
    ("random", 1000): 0.5820,
    ("greedy", 1000): 0.5836,
    ("hybrid_stackelberg", 2000): 0.8260,
    ("stackelberg", 2000): 0.8110,
    ("dqn", 2000): 0.7760,
    ("random", 2000): 0.5520,
    ("greedy", 2000): 0.5530,
    ("hybrid_stackelberg", 4000): 0.8030,
    ("stackelberg", 4000): 0.7895,
    ("dqn", 4000): 0.7423,
    ("random", 4000): 0.5210,
    ("greedy", 4000): 0.5258,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize speed and result screening runs")
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    speed_rows = read_rows(args.speed)
    result_rows = read_rows(args.result)

    lines = [
        "# Optimization screening summary",
        "",
        "Speed profiles keep the physical workload unchanged. Result profiles keep the task "
        "workload unchanged and vary only thesis-unspecified channel or cloud parameters.",
        "",
        "## Speed ranking",
        "",
        "| Vehicles | Profile | Strategy | Wall (s) | Success | Estimation (s) | Policy (s) | Training (s) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        speed_rows,
        key=lambda item: (int(item["configured_vehicle_count"]), float(item["wall_clock_s"])),
    ):
        lines.append(
            f"| {row['configured_vehicle_count']} | {row['profile']} | {row['strategy']} | "
            f"{float(row['wall_clock_s']):.2f} | {100 * float(row['success_rate']):.2f}% | "
            f"{float(row['phase_estimation_s']):.2f} | {float(row['phase_policy_s']):.2f} | "
            f"{float(row['phase_dqn_training_s']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Result ranking",
            "",
            "| Vehicles | Profile | Success | Paper target (same scale) | Delta | Oracle | Wall (s) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        result_rows,
        key=lambda item: (int(item["configured_vehicle_count"]), -float(item["success_rate"])),
    ):
        vehicles = int(row["configured_vehicle_count"])
        target = PAPER_SUCCESS[(row["strategy"], vehicles)]
        success = float(row["success_rate"])
        lines.append(
            f"| {row['configured_vehicle_count']} | {row['profile']} | {100 * success:.2f}% | "
            f"{100 * target:.2f}% | {100 * (success - target):+.2f} pp | "
            f"{100 * float(row['oracle_success_rate']):.2f}% | {float(row['wall_clock_s']):.2f} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
