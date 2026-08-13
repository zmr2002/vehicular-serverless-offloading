from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_matrix_summary(input_file: str | Path, output_file: str | Path) -> Path:
    with Path(input_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("matrix summary is empty")
    strategies = sorted({row["strategy"] for row in rows})
    metrics = (
        ("success_rate_mean", "Success rate", "rate"),
        ("avg_latency_s_mean", "Average latency", "seconds"),
        ("avg_energy_j_mean", "Average energy", "joules"),
        ("total_cost_mean", "Total cost", "cost units"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        for strategy in strategies:
            selected = sorted(
                (row for row in rows if row["strategy"] == strategy),
                key=lambda row: int(row["vehicle_count"]),
            )
            x = [int(row["vehicle_count"]) for row in selected]
            y = [float(row[metric]) for row in selected]
            ci = [float(row.get(metric.replace("_mean", "_ci95"), 0.0)) for row in selected]
            axis.errorbar(x, y, yerr=ci, marker="o", capsize=3, label=strategy)
        axis.set_title(title)
        axis.set_xlabel("Configured vehicles")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
