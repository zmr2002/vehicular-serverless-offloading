from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor
import subprocess
from time import perf_counter, process_time
import tomllib

from .config import SimulationConfig
from .domain import Task
from .serverless import HttpKnativeBackend


METRICS = (
    "success_rate", "avg_energy_j", "avg_latency_s", "avg_success_latency_s",
    "total_cost", "avg_cost_per_task", "avg_reward",
    "local_offload_ratio", "v2v_offload_ratio", "v2i_offload_ratio",
    "local_success_rate", "v2v_success_rate", "v2i_success_rate",
    "oracle_success_rate", "avoidable_failure_rate", "avg_decision_regret_s",
    "avg_server_distance_m",
    "dqn_decision_ratio", "avg_allowed_action_count", "hybrid_deviation_ratio",
    "hybrid_beneficial_deviation_rate", "all_actions_late_rate",
    "avg_hybrid_game_evidence", "avg_hybrid_dqn_evidence",
    "avg_hybrid_q_opposition", "avg_hybrid_cloud_pressure",
    "all_late_cloud_admission_rate", "all_late_cloud_to_capacity_ratio",
    "dqn_deviation_ratio", "rule_deviation_ratio",
)


def run_matrix(matrix_file: str | Path) -> Path:
    from .simulation import SimulationRunner

    matrix_path = Path(matrix_file).resolve()
    with matrix_path.open("rb") as handle:
        raw = tomllib.load(handle)["experiment"]
    base_path = (matrix_path.parent / raw["base_config"]).resolve()
    base = SimulationConfig.from_toml(base_path)
    output_root = (matrix_path.parent.parent / raw.get("output_dir", "results/verified/matrix")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / "runs"
    completed = _load_completed_runs(run_root)
    rows: list[dict] = []
    # Group by vehicle scale so the first strategy records one SUMO trace and
    # the remaining strategies immediately reuse it before moving to the next
    # (potentially much denser) traffic scenario.
    for vehicles in raw["vehicle_counts"]:
        for strategy in raw["strategies"]:
            for seed in raw["seeds"]:
                config = copy.deepcopy(base)
                config.strategy = strategy
                config.vehicle_count = int(vehicles)
                config.seed = int(seed)
                config.steps = int(raw.get("steps", config.steps))
                config.output_dir = str(run_root)
                config.validate()
                signature = _config_signature(config)
                key = (strategy, int(vehicles), int(seed), config.steps, config.backend, config.mobility, signature)
                row = completed.get(key)
                if row is None:
                    wall_started = perf_counter()
                    cpu_started = process_time()
                    summary, run_dir = SimulationRunner(config).run()
                    wall_clock_s = perf_counter() - wall_started
                    process_cpu_s = process_time() - cpu_started
                    row = asdict(summary)
                    row["run_dir"] = run_dir
                    row["matrix_config_signature"] = signature
                    row["wall_clock_s"] = wall_clock_s
                    row["process_cpu_s"] = process_cpu_s
                    timing_path = Path(run_dir) / "timing.json"
                    _append_process_timing(timing_path, wall_clock_s, process_cpu_s)
                    _include_timing_columns(row, timing_path)
                rows.append(row)
                _write_rows(output_root / "matrix-progress.csv", rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detail_path = output_root / f"matrix-detail-{timestamp}.csv"
    _write_rows(detail_path, rows)
    aggregate = _aggregate(rows)
    _write_rows(output_root / f"matrix-summary-{timestamp}.csv", aggregate)
    return detail_path


def _config_signature(config: SimulationConfig | dict) -> str:
    value = config.to_dict() if isinstance(config, SimulationConfig) else copy.deepcopy(config)
    value.pop("output_dir", None)
    value.pop("scenario_config", None)
    value["_code_commit"] = _git_commit()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def _load_completed_runs(run_root: Path) -> dict[tuple, dict]:
    completed: dict[tuple, dict] = {}
    if not run_root.exists():
        return completed
    for summary_path in sorted(run_root.glob("*/summary.json")):
        config_path = summary_path.with_name("config.json")
        environment_path = summary_path.with_name("environment.json")
        if not config_path.exists() or not environment_path.exists():
            continue
        try:
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        current_commit = _git_commit()
        if current_commit and saved_environment.get("git_commit") != current_commit:
            continue
        signature = _config_signature(saved_config)
        key = (
            row.get("strategy"),
            row.get("configured_vehicle_count"),
            row.get("seed"),
            row.get("configured_steps"),
            row.get("backend"),
            row.get("mobility"),
            signature,
        )
        row["run_dir"] = str(summary_path.parent)
        row["matrix_config_signature"] = signature
        _include_timing_columns(row, summary_path.with_name("timing.json"))
        completed[key] = row
    return completed


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _include_timing_columns(row: dict, timing_path: Path) -> None:
    if not timing_path.exists():
        return
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    row.setdefault("wall_clock_s", timing.get("wall_clock_s"))
    if "process_cpu_s" in timing:
        row["process_cpu_s"] = timing["process_cpu_s"]
    for phase, seconds in timing.get("phase_seconds", {}).items():
        row[f"phase_{phase}_s"] = seconds


def _append_process_timing(timing_path: Path, wall_clock_s: float, process_cpu_s: float) -> None:
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        timing = {}
    timing["wall_clock_s"] = wall_clock_s
    timing["process_cpu_s"] = process_cpu_s
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")


def benchmark_serverless(
    endpoint: str,
    output_dir: str | Path,
    concurrencies: tuple[int, ...] = (1, 10, 50),
    requests_per_level: int = 50,
    work_units: int = 25_000,
) -> Path:
    if work_units < 1 or work_units > 1_000_000:
        raise ValueError("work_units must be between 1 and 1,000,000")
    config = SimulationConfig()
    config.serverless.endpoint = endpoint
    config.serverless.max_work_units = work_units
    backend = HttpKnativeBackend(config.serverless)
    rows: list[dict] = []

    def invoke(index: int, level: int, phase: str) -> dict:
        task = Task(
            task_id=f"bench-{phase}-{level}-{index}",
            vehicle_id="benchmark",
            compute_cycles=float(work_units) * 1e6,
            data_size_mb=1.0,
            deadline_s=10.0,
            urgency=0.5,
            created_step=0,
        )
        measured = backend.execute(task, 0, index)
        return {
            "phase": phase,
            "concurrency": level,
            "request_index": index,
            "work_units": work_units,
            "dispatch_queue_ms": measured.dispatch_queue_ms,
            "http_latency_ms": measured.http_latency_ms,
            "client_latency_ms": measured.client_latency_ms,
            "processing_ms": measured.processing_ms,
            "platform_overhead_ms": measured.platform_overhead_ms,
            "http_attempts": measured.http_attempts,
            "http_retry_count": measured.http_retry_count,
            "retry_backoff_ms": measured.retry_backoff_ms,
            "cold_start": int(measured.cold_start),
            "instance_id": measured.instance_id,
            "checksum": measured.checksum,
        }

    try:
        # Do not probe health before these calls: after scale-to-zero, that probe would
        # start a Revision and silently remove the platform cold-start delay.
        rows.append(invoke(0, 1, "cold"))
        rows.append(invoke(1, 1, "warm"))
        health = backend.health()
        for level in concurrencies:
            with ThreadPoolExecutor(max_workers=level) as pool:
                rows.extend(
                    pool.map(lambda index: invoke(index, level, "burst"), range(requests_per_level))
                )
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output / f"serverless-benchmark-{timestamp}.csv"
    _write_rows(csv_path, rows)
    (output / f"serverless-benchmark-{timestamp}.json").write_text(
        json.dumps(
            {
                "endpoint": endpoint,
                "concurrencies": list(concurrencies),
                "requests_per_level": requests_per_level,
                "work_units": work_units,
                "health_after_cold_and_warm": health,
                "cold_request": rows[0],
                "warm_request": rows[1],
                "requests": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["strategy"], row["configured_vehicle_count"]), []).append(row)
    output: list[dict] = []
    for (strategy, vehicles), group in sorted(groups.items()):
        result: dict = {"strategy": strategy, "vehicle_count": vehicles, "runs": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            deviation = stdev(values) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = deviation
            result[f"{metric}_ci95"] = 1.96 * deviation / math.sqrt(len(values)) if values else 0.0
        output.append(result)
    return output


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result set")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
