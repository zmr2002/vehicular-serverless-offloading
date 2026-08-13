from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


GROUPS = (
    ("baselines", None),
    ("thesis-hybrid", "thesis_hybrid"),
    ("enhanced-hybrid", "enhanced_hybrid"),
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "results" / "verified" / "final-model-comparison"
    rows: list[dict[str, str]] = []
    sources = []
    for directory_name, model_override in GROUPS:
        source = _latest_complete_session(root / directory_name)
        sources.append(source)
        with (source / "evaluation-results.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                row["model"] = model_override or row["strategy"]
                rows.append(row)
    rows.sort(key=lambda row: (int(row["configured_vehicle_count"]), row["model"]))
    fieldnames = ["model"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (root / "comparison-results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Final single-seed model comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Vehicles | Model | Success | Latency | Energy | Cost | "
        "Local | V2V | V2I | DQN decisions | Strict dominance | Game gate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['configured_vehicle_count']} | {row['model']} | "
            f"{float(row['success_rate']):.2%} | "
            f"{float(row['avg_latency_s']):.3f} s | "
            f"{float(row['avg_energy_j']):.2f} J | "
            f"{float(row['avg_cost_per_task']):.3f} | "
            f"{float(row['local_offload_ratio']):.2%} | "
            f"{float(row['v2v_offload_ratio']):.2%} | "
            f"{float(row['v2i_offload_ratio']):.2%} | "
            f"{float(row['dqn_decision_ratio']):.2%} | "
            f"{float(row.get('hybrid_strict_dominance_ratio') or 0.0):.2%} | "
            f"{float(row.get('hybrid_game_gate_ratio') or 0.0):.2%} |"
        )
    markdown.extend(
        [
            "",
            "Source sessions:",
            *[f"- `{source}`" for source in sources],
            "",
        ]
    )
    summary = root / "comparison-summary.md"
    summary.write_text("\n".join(markdown), encoding="utf-8")
    print(f"COMPLETE {root}")
    print(f"SUMMARY {summary}")
    return 0


def _latest_complete_session(group: Path) -> Path:
    candidates = [
        path
        for path in group.glob("run-*")
        if (path / "evaluation-results.csv").exists()
        and (path / "diagnostic-summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no complete pipeline session found in {group}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


if __name__ == "__main__":
    raise SystemExit(main())
