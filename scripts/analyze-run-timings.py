from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def analyze_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    action_counts: Counter[str] = Counter()
    action_successes: Counter[str] = Counter()
    path_nodes: defaultdict[str, int] = defaultdict(int)
    oracle_actions: Counter[str] = Counter()
    with (run_dir / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = row["action"]
            action_counts[action] += 1
            action_successes[action] += int(row["success"])
            path_nodes[action] += len(row["path"].split(">")) if row["path"] else 0
            oracle_actions[row.get("oracle_action", "unknown")] += 1
    actions = {
        action: {
            "count": count,
            "ratio": count / summary["total_tasks"],
            "success_rate": action_successes[action] / count,
            "avg_path_nodes": path_nodes[action] / count,
        }
        for action, count in sorted(action_counts.items())
    }
    return {
        "run_dir": str(run_dir),
        "strategy": summary["strategy"],
        "vehicles": summary["configured_vehicle_count"],
        "tasks": summary["total_tasks"],
        "success_rate": summary["success_rate"],
        "wall_clock_s": timing["wall_clock_s"],
        "process_cpu_s": timing.get("process_cpu_s"),
        "phase_seconds": timing["phase_seconds"],
        "segments": [
            {
                **segment,
                "ms_per_task": 1000.0 * segment["wall_s"] / segment["tasks"]
                if segment["tasks"]
                else 0.0,
            }
            for segment in timing["segments"]
        ],
        "actions": actions,
        "oracle_actions": dict(sorted(oracle_actions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream task logs and report per-run timing diagnostics")
    parser.add_argument("root", type=Path)
    parser.add_argument("--vehicles", type=int)
    parser.add_argument("--strategy", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = set(args.strategy or [])
    reports = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if args.vehicles is not None and summary["configured_vehicle_count"] != args.vehicles:
            continue
        if selected and summary["strategy"] not in selected:
            continue
        reports.append(analyze_run(summary_path.parent))
    encoded = json.dumps(reports, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
