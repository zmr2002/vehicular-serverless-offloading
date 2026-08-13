from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tomllib

import numpy as np


def load_positions(path: Path, frame_stride: int) -> np.ndarray:
    points: list[tuple[float, float]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("format") != 1:
            raise ValueError(f"unsupported mobility trace format: {header}")
        for frame_index, line in enumerate(handle):
            if frame_index % frame_stride:
                continue
            payload = json.loads(line)
            points.extend((float(vehicle[1]), float(vehicle[2])) for vehicle in payload[2])
    if not points:
        raise ValueError(f"trace contains no sampled positions: {path}")
    return np.asarray(points, dtype=np.float64)


def nearest_squared(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = squared.argmin(axis=1)
    return labels, squared[np.arange(len(points)), labels]


def optimize(points: np.ndarray, stations: int, restarts: int, seed: int) -> np.ndarray:
    best_centers: np.ndarray | None = None
    best_objective = float("inf")
    master = np.random.default_rng(seed)
    for _ in range(restarts):
        rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        centers = [points[int(rng.integers(0, len(points)))]]
        while len(centers) < stations:
            _, distances = nearest_squared(points, np.asarray(centers))
            total = distances.sum()
            index = int(rng.choice(len(points), p=distances / total)) if total else 0
            centers.append(points[index])
        candidate = np.asarray(centers, dtype=np.float64)
        for _iteration in range(100):
            labels, distances = nearest_squared(points, candidate)
            updated = candidate.copy()
            for cluster in range(stations):
                members = points[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = points[int(distances.argmax())]
            if np.allclose(updated, candidate, atol=0.01):
                candidate = updated
                break
            candidate = updated
        _, distances = nearest_squared(points, candidate)
        objective = float(distances.mean())
        if objective < best_objective:
            best_objective = objective
            best_centers = candidate
    assert best_centers is not None
    return best_centers[np.lexsort((best_centers[:, 1], best_centers[:, 0]))]


def coverage(points: np.ndarray, centers: np.ndarray) -> dict:
    _, squared = nearest_squared(points, centers)
    distances = np.sqrt(squared)
    return {
        "samples": len(points),
        "mean_m": float(distances.mean()),
        "median_m": float(np.quantile(distances, 0.50)),
        "p90_m": float(np.quantile(distances, 0.90)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "p99_m": float(np.quantile(distances, 0.99)),
        "max_m": float(distances.max()),
        "within_500m": float((distances <= 500.0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate base stations on one trace and evaluate another")
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--evaluation-trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.toml"))
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        baseline = np.asarray(tomllib.load(handle)["simulation"]["service_positions"], dtype=np.float64)
    calibration = load_positions(args.calibration_trace, args.frame_stride)
    evaluation = load_positions(args.evaluation_trace, args.frame_stride)
    optimized = optimize(calibration, args.stations, args.restarts, args.seed)
    rounded = np.round(optimized, 2)
    report = {
        "method": "deterministic multi-start k-means on an independent mobility trace",
        "calibration_trace": str(args.calibration_trace),
        "evaluation_trace": str(args.evaluation_trace),
        "frame_stride": args.frame_stride,
        "baseline_positions": baseline.tolist(),
        "optimized_positions": rounded.tolist(),
        "calibration": {
            "baseline": coverage(calibration, baseline),
            "optimized": coverage(calibration, rounded),
        },
        "evaluation": {
            "baseline": coverage(evaluation, baseline),
            "optimized": coverage(evaluation, rounded),
        },
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
