"""Select the final-model candidate from the screening arms.

Pre-declared rule: among the candidate arms, maximize the WORST vehicle-scale
mean success margin over the best baseline; break ties by the overall mean
margin. The damping-only arm is an ablation and never selected. Prints the
winning driver arm name as the last stdout line.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPO / "results" / "verified" / "hybrid-cross-checkpoint-eval" / "arms-summary.csv"
)
CANDIDATES = {
    "dqnckpt-adq": "dqnckpt-adequacy",
    "dqnckpt-cap": "dqnckpt-cap",
}
EXPECTED_SCALES = {1000, 2000, 4000}
EXPECTED_REPLICATES = 3


def main() -> int:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    scores = {}
    for arm_suffix, arm_name in CANDIDATES.items():
        arm_rows = [row for row in rows if row["arm"] == arm_suffix]
        by_scale: dict[int, list[float]] = {}
        for row in arm_rows:
            by_scale.setdefault(int(row["vehicles"]), []).append(
                float(row["margin_over_best_baseline"])
            )
        if set(by_scale) != EXPECTED_SCALES or any(
            len(values) != EXPECTED_REPLICATES for values in by_scale.values()
        ):
            raise SystemExit(
                f"arm {arm_suffix} is incomplete: "
                f"{ {scale: len(values) for scale, values in by_scale.items()} }"
            )
        scale_margins = {scale: fmean(values) for scale, values in by_scale.items()}
        worst = min(scale_margins.values())
        overall = fmean(scale_margins.values())
        scores[arm_name] = (worst, overall)
        print(
            f"candidate {arm_name}: per-scale margins "
            + ", ".join(
                f"{scale}: {100 * margin:+.2f} pp"
                for scale, margin in sorted(scale_margins.items())
            )
            + f"; worst {100 * worst:+.2f} pp, mean {100 * overall:+.2f} pp"
        )
    winner = max(scores, key=lambda name: scores[name])
    print(f"selected {winner} by pre-declared worst-scale rule")
    print(winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
