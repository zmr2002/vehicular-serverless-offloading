from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
from hashlib import sha256
import json
from itertools import zip_longest
from pathlib import Path
import shutil
import statistics
import subprocess
from threading import Event, Thread
from time import monotonic, sleep
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.serverless import SERVERLESS_DELAY_MODEL
from vehicular_offloading.simulation import SimulationRunner


BYTES_PER_RAW_TASK_RECORD = 2_500
BYTES_PER_RAW_DECISION_TRACE = 160


@dataclass(slots=True, frozen=True)
class ValidationCase:
    vehicle_count: int
    steps: int
    seed: int
    checkpoint: Path
    request_budget: int

    @property
    def key_prefix(self) -> str:
        return f"v{self.vehicle_count}:s{self.steps}:seed{self.seed}"

    @property
    def output_name(self) -> str:
        return (
            f"vehicles-{self.vehicle_count}-steps-{self.steps}-seed-{self.seed}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--profile", default="knative")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--vehicles", type=int, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analytical-only", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        settings = tomllib.load(handle)["validation"]
    base_path = _resolve(repo / "configs", settings["base_config"])
    steps = tuple(args.steps or settings["steps"])
    if not steps or any(value <= 0 for value in steps):
        raise ValueError("validation steps must contain positive integers")
    configured_vehicles = (
        settings["vehicle_counts"]
        if "vehicle_counts" in settings
        else [settings["vehicle_count"]]
    )
    vehicle_counts = tuple(args.vehicles or configured_vehicles)
    if not vehicle_counts or any(value <= 0 for value in vehicle_counts):
        raise ValueError("validation vehicle counts must contain positive integers")
    seeds = tuple(
        int(value)
        for value in settings.get("seeds", [settings.get("seed", 48)])
    )
    if not seeds or any(value < 0 for value in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("validation seeds must be unique non-negative integers")
    modes = tuple(settings.get("modes", ["analytical", "knative"]))
    allowed_modes = {"analytical", "knative", "knative_replay", "knative_closed_loop"}
    if not modes or modes[0] != "analytical" or set(modes) - allowed_modes:
        raise ValueError(
            "validation modes must start with analytical and contain only "
            "analytical, knative, knative_replay, or knative_closed_loop"
        )
    if args.analytical_only:
        modes = ("analytical",)
    cases = _validation_cases(
        repo,
        settings,
        vehicle_counts,
        steps,
        args.checkpoint,
        seeds,
    )
    task_probability = SimulationConfig.from_toml(base_path).task_probability
    minimum_free_disk_gb = float(settings["minimum_free_disk_gb"])
    output_root = _resolve(repo, settings["output_dir"])

    signature = _signature(
        config_path,
        base_path,
        tuple(case.checkpoint for case in cases),
        args.endpoint,
        cases,
        modes,
    )
    commit = _git_commit(repo)
    session = (
        args.session.resolve()
        if args.session is not None
        else output_root / f"run-{commit[:8]}-{signature[:12]}"
    )
    try:
        session.relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"resume session must stay inside {output_root.resolve()}: {session}"
        ) from error
    session.mkdir(parents=True, exist_ok=True)
    state_path = session / "state.json"
    state = _read_json(state_path, {"signature": signature, "runs": {}})
    if state.get("signature") != signature:
        raise RuntimeError(f"existing state has a different signature: {state_path}")
    execution_commits = state.setdefault("execution_commits", [])
    provenance_changed = False
    for saved in state["runs"].values():
        environment = _read_json(Path(saved["run_dir"]) / "environment.json", {})
        saved_commit = environment.get("git_commit")
        if saved_commit and saved_commit not in execution_commits:
            execution_commits.append(saved_commit)
            provenance_changed = True
    if commit not in execution_commits:
        execution_commits.append(commit)
        provenance_changed = True
    if provenance_changed:
        _write_json_atomic(state_path, state)

    task_count_estimates = settings.get("estimated_task_counts", {})
    estimate_reference_steps = int(
        settings.get("estimated_task_count_steps", max(steps))
    )
    estimated_task_counts = [
        int(
            (
                int(task_count_estimates[str(case.vehicle_count)])
                * case.steps
                / estimate_reference_steps
            )
            if str(case.vehicle_count) in task_count_estimates
            else case.steps * case.vehicle_count * task_probability
        )
        for case in cases
    ]
    estimated_raw_gb = _estimated_storage_gb(
        estimated_task_counts,
        modes,
        float(settings["task_record_sample_rate"]),
    )
    free_gb = shutil.disk_usage(session).free / 1024**3
    preflight = {
        "created_at": _utc_now(),
        "endpoint": args.endpoint,
        "profile": args.profile,
        "steps": list(steps),
        "vehicle_counts": list(vehicle_counts),
        "seeds": list(seeds),
        "modes": list(modes),
        "strategy": settings["strategy"],
        "cases": [
            {
                "vehicle_count": case.vehicle_count,
                "steps": case.steps,
                "seed": case.seed,
                "checkpoint": str(case.checkpoint),
                "checkpoint_sha256": _file_sha256(case.checkpoint),
                "request_budget": case.request_budget,
            }
            for case in cases
        ],
        "client_concurrency": int(
            settings.get(
                "client_concurrency",
                SimulationConfig.from_toml(base_path).serverless.client_concurrency,
            )
        ),
        "task_record_sample_rate": float(settings["task_record_sample_rate"]),
        "estimated_upper_bound_raw_gb": estimated_raw_gb,
        "free_disk_gb": free_gb,
        "minimum_free_disk_gb": minimum_free_disk_gb,
    }
    _write_json_atomic(session / "preflight.json", preflight)
    if free_gb - estimated_raw_gb < minimum_free_disk_gb:
        raise RuntimeError(
            f"storage preflight failed: {free_gb:.2f} GiB free, "
            f"{estimated_raw_gb:.2f} GiB estimated raw output, "
            f"{minimum_free_disk_gb:.2f} GiB reserve"
        )

    print(
        f"SESSION {session}\n"
        f"Storage upper bound {estimated_raw_gb:.2f} GiB; "
        f"free {free_gb:.2f} GiB; reserve {minimum_free_disk_gb:.2f} GiB.",
        flush=True,
    )
    if args.preflight_only:
        reusable = [
            key
            for key, saved in state["runs"].items()
            if _completed_run(Path(saved["run_dir"]))
        ]
        if reusable:
            print(f"RESUME completed={','.join(sorted(reusable))}", flush=True)
        for case in cases:
            print(
                f"PREFLIGHT vehicles={case.vehicle_count} steps={case.steps} "
                f"seed={case.seed} "
                f"request_budget={case.request_budget} "
                f"checkpoint={case.checkpoint}",
                flush=True,
            )
        print("PREFLIGHT COMPLETE", flush=True)
        return 0

    _run_analytical_phase(
        repo=repo,
        base_path=base_path,
        cases=cases,
        endpoint=args.endpoint,
        session=session,
        state=state,
        state_path=state_path,
        settings=settings,
        modes=modes,
    )
    for stage_index, case in enumerate(cases, start=1):
        trace_path = session / "decision-traces" / f"{case.output_name}.jsonl.gz"
        analytical = state["runs"][f"{case.key_prefix}:analytical"]
        predicted_requests = round(
            int(analytical["total_tasks"]) * float(analytical["v2i_offload_ratio"])
        )
        if predicted_requests > case.request_budget:
            raise RuntimeError(
                f"{case.vehicle_count}-vehicle/{case.steps}-step analytical run "
                f"predicts {predicted_requests} V2I requests, above "
                f"request_budget={case.request_budget}. "
                "Review the completed analytical result before raising the budget."
            )
        print(
            f"STAGE {stage_index}/{len(cases)} vehicles={case.vehicle_count} "
            f"steps={case.steps}: "
            f"analytical predicts {predicted_requests} live requests.",
            flush=True,
        )
        for mode in modes[1:]:
            _run_or_resume(
                repo=repo,
                base_path=base_path,
                case=case,
                endpoint=args.endpoint,
                session=session,
                state=state,
                state_path=state_path,
                settings=settings,
                backend="knative",
                run_mode=mode,
                profile=args.profile,
                decision_trace_mode=(
                    "replay" if mode == "knative_replay" else "none"
                ),
                decision_trace_path=trace_path,
            )
        _write_comparisons(session, state, cases)

    summary_path = _write_comparisons(session, state, cases)
    print(f"COMPLETE {session}", flush=True)
    print(f"SUMMARY {summary_path}", flush=True)
    return 0


def _run_analytical_phase(
    *,
    repo: Path,
    base_path: Path,
    cases: tuple[ValidationCase, ...],
    endpoint: str,
    session: Path,
    state: dict,
    state_path: Path,
    settings: dict,
    modes: tuple[str, ...],
) -> None:
    pending = []
    trace_mode = "record" if "knative_replay" in modes else "none"
    for case in cases:
        key = f"{case.key_prefix}:analytical"
        trace_path = session / "decision-traces" / f"{case.output_name}.jsonl.gz"
        saved = state["runs"].get(key)
        trace_ready = trace_mode == "none" or trace_path.is_file()
        if saved and _completed_run(Path(saved["run_dir"])) and trace_ready:
            print(
                f"SKIP vehicles={case.vehicle_count} steps={case.steps} "
                f"seed={case.seed} mode=analytical",
                flush=True,
            )
            continue
        config = _validation_config(
            base_path=base_path,
            case=case,
            endpoint=endpoint,
            session=session,
            settings=settings,
            backend="analytical",
            run_mode="analytical",
            decision_trace_mode=trace_mode,
            decision_trace_path=trace_path,
        )
        pending.append((case, key, config))
    if not pending:
        return
    requested = int(settings.get("analytical_parallelism", 1))
    if requested <= 0:
        raise ValueError("analytical_parallelism must be positive")
    workers = min(requested, len(pending))
    print(
        f"ANALYTICAL PHASE: {len(pending)} pending cases, "
        f"parallelism={workers}.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for case, key, config in sorted(
            pending,
            key=lambda item: item[0].vehicle_count,
            reverse=True,
        ):
            print(
                f"START vehicles={case.vehicle_count} steps={case.steps} "
                f"seed={case.seed} mode=analytical",
                flush=True,
            )
            futures[executor.submit(_execute_validation_config, config, "analytical")] = (
                case,
                key,
            )
        failures = []
        for future in as_completed(futures):
            case, key = futures[future]
            try:
                row = future.result()
            except Exception as error:
                failures.append((case, error))
                print(
                    f"FAILED vehicles={case.vehicle_count} seed={case.seed} "
                    f"mode=analytical: {error}",
                    flush=True,
                )
                continue
            state["runs"][key] = row
            _write_json_atomic(state_path, state)
            print(
                f"DONE vehicles={case.vehicle_count} steps={case.steps} "
                f"seed={case.seed} mode=analytical tasks={row['total_tasks']} "
                f"v2i={float(row['v2i_offload_ratio']):.2%} "
                f"success={float(row['success_rate']):.2%} "
                f"wall={float(row['wall_clock_s']):.1f}s",
                flush=True,
            )
        if failures:
            case, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} analytical validation case(s) failed; "
                f"first was vehicles={case.vehicle_count} seed={case.seed}"
            ) from error


def _run_or_resume(
    *,
    repo: Path,
    base_path: Path,
    case: ValidationCase,
    endpoint: str,
    session: Path,
    state: dict,
    state_path: Path,
    settings: dict,
    backend: str,
    run_mode: str,
    profile: str,
    decision_trace_mode: str = "none",
    decision_trace_path: Path | None = None,
) -> dict:
    key = f"{case.key_prefix}:{run_mode}"
    saved = state["runs"].get(key)
    if saved and _completed_run(Path(saved["run_dir"])):
        print(
            f"SKIP vehicles={case.vehicle_count} steps={case.steps} "
            f"mode={run_mode}",
            flush=True,
        )
        return saved

    config = _validation_config(
        base_path=base_path,
        case=case,
        endpoint=endpoint,
        session=session,
        settings=settings,
        backend=backend,
        run_mode=run_mode,
        decision_trace_mode=decision_trace_mode,
        decision_trace_path=decision_trace_path,
    )

    timeout_s = float(settings["scale_to_zero_timeout_s"])
    poll_s = float(settings["pod_poll_interval_s"])
    sampler = None
    if backend == "knative":
        _wait_for_zero(profile, timeout_s, poll_s)
        timeline = (
            session / "runs" / case.output_name / f"{run_mode}-pods.csv"
        )
        timeline.parent.mkdir(parents=True, exist_ok=True)
        sampler = PodSampler(profile, timeline, poll_s)
        sampler.start()

    print(
        f"START vehicles={case.vehicle_count} steps={case.steps} "
        f"seed={case.seed} mode={run_mode}",
        flush=True,
    )
    try:
        row = _execute_validation_config(config, run_mode)
        if backend == "knative":
            _wait_for_zero(profile, timeout_s, poll_s)
    finally:
        if sampler is not None:
            sampler.stop()

    state["runs"][key] = row
    _write_json_atomic(state_path, state)
    print(
        f"DONE vehicles={case.vehicle_count} steps={case.steps} "
        f"seed={case.seed} mode={run_mode} tasks={row['total_tasks']} "
        f"v2i={float(row['v2i_offload_ratio']):.2%} "
        f"success={float(row['success_rate']):.2%} "
        f"wall={float(row['wall_clock_s']):.1f}s",
        flush=True,
    )
    return row


def _validation_config(
    *,
    base_path: Path,
    case: ValidationCase,
    endpoint: str,
    session: Path,
    settings: dict,
    backend: str,
    run_mode: str,
    decision_trace_mode: str,
    decision_trace_path: Path | None,
) -> SimulationConfig:
    config = SimulationConfig.from_toml(base_path)
    config.steps = case.steps
    config.vehicle_count = case.vehicle_count
    config.seed = case.seed
    config.strategy = str(settings["strategy"])
    config.backend = backend
    config.output_dir = str(session / "runs" / case.output_name / run_mode)
    config.record_task_records = True
    config.task_record_sample_rate = float(settings["task_record_sample_rate"])
    config.minimum_free_disk_gb = float(settings["minimum_free_disk_gb"])
    config.dqn.mode = "evaluate"
    config.dqn.checkpoint_path = str(case.checkpoint)
    config.serverless.endpoint = endpoint
    config.serverless.client_concurrency = int(
        settings.get("client_concurrency", config.serverless.client_concurrency)
    )
    config.serverless.max_requests_per_run = (
        case.request_budget if backend == "knative" else 0
    )
    config.decision_trace_mode = decision_trace_mode
    config.decision_trace_path = (
        str(decision_trace_path) if decision_trace_mode != "none" else None
    )
    config.validate()
    return config


def _execute_validation_config(
    config: SimulationConfig,
    run_mode: str,
) -> dict:
    started = monotonic()
    summary, run_dir = SimulationRunner(config).run()
    wall_s = monotonic() - started
    archive = _compress_task_records(Path(run_dir))
    return {
        **asdict(summary),
        "run_mode": run_mode,
        "wall_clock_s": wall_s,
        "run_dir": run_dir,
        "task_record_archive": str(archive),
    }


class PodSampler:
    def __init__(self, profile: str, output: Path, interval_s: float):
        self.profile = profile
        self.output = output
        self.interval_s = max(0.2, interval_s)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="knative-pod-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_s * 2))

    def _run(self) -> None:
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "timestamp_utc",
                    "pod_count",
                    "ready_pod_count",
                    "pod_names",
                    "error",
                ),
            )
            writer.writeheader()
            while not self._stop.is_set():
                try:
                    pods = _workload_pods(self.profile)
                    ready = sum(
                        any(
                            condition.get("type") == "Ready"
                            and condition.get("status") == "True"
                            for condition in item.get("status", {}).get("conditions", [])
                        )
                        for item in pods
                    )
                    writer.writerow(
                        {
                            "timestamp_utc": _utc_now(),
                            "pod_count": len(pods),
                            "ready_pod_count": ready,
                            "pod_names": ";".join(
                                item.get("metadata", {}).get("name", "") for item in pods
                            ),
                            "error": "",
                        }
                    )
                except Exception as error:
                    writer.writerow(
                        {
                            "timestamp_utc": _utc_now(),
                            "pod_count": "",
                            "ready_pod_count": "",
                            "pod_names": "",
                            "error": str(error),
                        }
                    )
                handle.flush()
                self._stop.wait(self.interval_s)


def _wait_for_zero(profile: str, timeout_s: float, poll_s: float) -> None:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if not _workload_pods(profile):
            return
        sleep(max(0.2, poll_s))
    raise TimeoutError(
        f"Knative service did not scale to zero within {timeout_s:.0f} seconds"
    )


def _workload_pods(profile: str) -> list[dict]:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            profile,
            "get",
            "pods",
            "-l",
            "serving.knative.dev/service=vehicular-task-function",
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout).get("items", [])


def _write_comparisons(
    session: Path,
    state: dict,
    cases: tuple[ValidationCase, ...],
) -> Path:
    rows = []
    details = {}
    for case in cases:
        analytical = state["runs"].get(f"{case.key_prefix}:analytical")
        if not analytical:
            continue
        for mode in ("knative_replay", "knative_closed_loop", "knative"):
            knative = state["runs"].get(f"{case.key_prefix}:{mode}")
            if not knative:
                continue
            comparison = _compare_runs(
                Path(analytical["run_dir"]),
                Path(knative["run_dir"]),
            )
            comparison.update(
                _aggregate_serverless_metrics(knative, comparison)
            )
            row = {
                "vehicle_count": case.vehicle_count,
                "steps": case.steps,
                "seed": case.seed,
                "mode": mode,
                "tasks": int(knative["total_tasks"]),
                "analytical_success_rate": float(analytical["success_rate"]),
                "knative_success_rate": float(knative["success_rate"]),
                "success_delta": float(knative["success_rate"])
                - float(analytical["success_rate"]),
                "analytical_v2i_ratio": float(analytical["v2i_offload_ratio"]),
                "knative_v2i_ratio": float(knative["v2i_offload_ratio"]),
                "analytical_wall_s": float(analytical["wall_clock_s"]),
                "knative_wall_s": float(knative["wall_clock_s"]),
                **comparison,
                **_pod_metrics(
                    session
                    / "runs"
                    / case.output_name
                    / f"{mode}-pods.csv"
                ),
            }
            rows.append(row)
            details[f"{case.key_prefix}:{mode}"] = row

    csv_path = session / "comparison-summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_json_atomic(session / "comparison-summary.json", details)
    markdown = [
        "# Analytical and live Knative validation",
        "",
        "| Vehicles | Steps | Seed | Mode | Tasks | Analytical success | Knative success | Delta | "
        "HTTP requests | V2I failures | Retried requests | HTTP retries | "
        "Cold maximum | Warm P95 | "
        "Dispatch P95 | HTTP P95 | Platform P95 | Physical compute | "
        "Scaled processing | Decomposition error | Pods | Sampled rows | "
        "Action changes (sample) |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['vehicle_count']} | {row['steps']} | {row['seed']} | "
            f"{row['mode']} | {row['tasks']} | "
            f"{row['analytical_success_rate']:.2%} | "
            f"{row['knative_success_rate']:.2%} | "
            f"{row['success_delta']:+.2%} | "
            f"{row['live_v2i_requests']} | {row['live_v2i_failures']} | "
            f"{row['retried_requests']} | {row['total_http_retries']} | "
            f"{row['cold_client_latency_max_ms']:.2f} ms | "
            f"{row['warm_client_latency_p95_ms']:.2f} ms | "
            f"{row['dispatch_queue_p95_ms']:.2f} ms | "
            f"{row['http_latency_p95_ms']:.2f} ms | "
            f"{row['platform_overhead_p95_ms']:.2f} ms | "
            f"{row['physical_compute_mean_ms']:.2f} ms | "
            f"{row['scaled_processing_mean_ms']:.2f} ms | "
            f"{row['delay_decomposition_max_error_ms']:.3g} ms | "
            f"{row['max_pods']} | "
            f"{row['matched_task_records']} | "
            f"{row['action_change_rate']:.2%} |"
        )
    markdown.extend(
        [
            "",
            "Replay mode freezes the analytical action trace and isolates backend "
            "overhead. Closed-loop mode allows measured Knative overhead to change "
            "the 20-dimensional state and later actions.",
            "",
        ]
    )
    summary_path = session / "comparison-summary.md"
    summary_path.write_text("\n".join(markdown), encoding="utf-8")
    return summary_path


def _aggregate_serverless_metrics(summary: dict, sampled: dict) -> dict:
    if "serverless_http_request_count" not in summary:
        return sampled
    return {
        "live_v2i_requests": int(summary["serverless_http_request_count"]),
        "live_v2i_failures": int(summary["serverless_v2i_failure_count"]),
        "retried_requests": int(summary["serverless_retried_request_count"]),
        "total_http_retries": int(summary["serverless_http_retry_count"]),
        "cold_start_flags": int(summary["serverless_cold_start_count"]),
        "distinct_instances": int(summary["serverless_distinct_instance_count"]),
        "client_latency_mean_ms": float(
            summary["avg_serverless_client_latency_ms"]
        ),
        "client_latency_p95_ms": float(
            summary["p95_serverless_client_latency_ms"]
        ),
        "client_latency_max_ms": float(
            summary["max_serverless_client_latency_ms"]
        ),
        "cold_client_latency_max_ms": float(
            summary["max_serverless_cold_client_latency_ms"]
        ),
        "warm_client_latency_p95_ms": float(
            summary["p95_serverless_warm_client_latency_ms"]
        ),
        "dispatch_queue_p95_ms": float(
            summary["p95_serverless_dispatch_queue_ms"]
        ),
        "http_latency_p95_ms": float(
            summary["p95_serverless_http_latency_ms"]
        ),
        "platform_overhead_p95_ms": float(
            summary["p95_serverless_platform_overhead_ms"]
        ),
        "physical_compute_mean_ms": float(
            summary["avg_serverless_physical_compute_ms"]
        ),
        "scaled_processing_mean_ms": float(
            summary["avg_serverless_scaled_processing_ms"]
        ),
        "delay_decomposition_max_error_ms": float(
            summary["serverless_delay_decomposition_max_error_ms"]
        ),
    }


def _compare_runs(analytical_dir: Path, knative_dir: Path) -> dict:
    for run_dir in (analytical_dir, knative_dir):
        environment = _read_json(run_dir / "environment.json", {})
        if environment.get("serverless_delay_model") != SERVERLESS_DELAY_MODEL:
            raise ValueError(
                f"run does not use {SERVERLESS_DELAY_MODEL}: {run_dir}"
            )
    matched = 0
    action_changes = 0
    live_v2i_requests = 0
    live_v2i_failures = 0
    cold_start_flags = 0
    retried_requests = 0
    total_http_retries = 0
    instances: set[str] = set()
    client = array("d")
    dispatch = array("d")
    http = array("d")
    processing = array("d")
    overhead = array("d")
    cold_client = array("d")
    warm_client = array("d")
    retry_backoff = array("d")
    physical_compute = array("d")
    physical_queue = array("d")
    scaled_processing = array("d")
    total_delay = array("d")
    decomposition_errors = array("d")

    for analytical, live in zip_longest(
        _task_rows(analytical_dir),
        _task_rows(knative_dir),
    ):
        if analytical is None or live is None:
            raise ValueError("analytical and Knative task record counts differ")
        if analytical["task_id"] != live["task_id"]:
            raise ValueError(
                "task record order differs: "
                f"{analytical['task_id']} != {live['task_id']}"
            )
        matched += 1
        action_changes += analytical["action"] != live["action"]
        if live["action"] != "v2i":
            continue
        live_v2i_requests += 1
        live_v2i_failures += not _truthy(live.get("success"))
        is_cold = _truthy(live.get("cold_start"))
        cold_start_flags += is_cold
        retries = _integer(live.get("http_retry_count"))
        retried_requests += retries > 0
        total_http_retries += retries
        _append_float(retry_backoff, live.get("retry_backoff_ms"))
        if live.get("instance_id"):
            instances.add(live["instance_id"])
        _append_float(client, live.get("client_latency_ms"))
        _append_float(dispatch, live.get("dispatch_queue_ms"))
        _append_float(http, live.get("http_latency_ms"))
        _append_float(processing, live.get("processing_ms"))
        _append_float(overhead, live.get("platform_overhead_ms"))
        _append_float(physical_compute, live.get("physical_compute_ms"))
        _append_float(physical_queue, live.get("physical_queue_ms"))
        _append_float(scaled_processing, live.get("scaled_processing_ms"))
        _append_float(total_delay, live.get("total_delay_ms"))
        components = [
            _optional_float(live.get("preprocessing_delay_ms")),
            _optional_float(live.get("radio_delay_ms")),
            _optional_float(live.get("physical_compute_ms")),
            _optional_float(live.get("physical_queue_ms")),
            _optional_float(live.get("dispatch_queue_ms")),
            _optional_float(live.get("platform_overhead_ms")),
        ]
        observed_total = _optional_float(live.get("total_delay_ms"))
        if observed_total is not None and all(value is not None for value in components):
            decomposition_errors.append(
                abs(observed_total - sum(value for value in components if value is not None))
            )
        if is_cold:
            _append_float(cold_client, live.get("client_latency_ms"))
        else:
            _append_float(warm_client, live.get("client_latency_ms"))

    maximum_decomposition_error = max(decomposition_errors, default=0.0)
    if maximum_decomposition_error > 1e-6:
        raise ValueError(
            "live V2I delay decomposition is inconsistent: "
            f"maximum error {maximum_decomposition_error:.9f} ms"
        )
    return {
        "serverless_delay_model": SERVERLESS_DELAY_MODEL,
        "matched_task_records": matched,
        "action_change_rate": action_changes / matched if matched else 0.0,
        "live_v2i_requests": live_v2i_requests,
        "live_v2i_failures": live_v2i_failures,
        "cold_start_flags": cold_start_flags,
        "retried_requests": retried_requests,
        "total_http_retries": total_http_retries,
        "retry_backoff_mean_ms": (
            statistics.fmean(retry_backoff) if retry_backoff else 0.0
        ),
        "retry_backoff_max_ms": max(retry_backoff, default=0.0),
        "distinct_instances": len(instances),
        "client_latency_mean_ms": statistics.fmean(client) if client else 0.0,
        "client_latency_p95_ms": _percentile(client, 0.95),
        "client_latency_max_ms": max(client, default=0.0),
        "dispatch_queue_mean_ms": statistics.fmean(dispatch) if dispatch else 0.0,
        "dispatch_queue_p95_ms": _percentile(dispatch, 0.95),
        "dispatch_queue_max_ms": max(dispatch, default=0.0),
        "http_latency_mean_ms": statistics.fmean(http) if http else 0.0,
        "http_latency_p95_ms": _percentile(http, 0.95),
        "http_latency_max_ms": max(http, default=0.0),
        "cold_client_latency_mean_ms": (
            statistics.fmean(cold_client) if cold_client else 0.0
        ),
        "cold_client_latency_max_ms": max(cold_client, default=0.0),
        "warm_client_latency_p95_ms": _percentile(warm_client, 0.95),
        "processing_mean_ms": statistics.fmean(processing) if processing else 0.0,
        "processing_p95_ms": _percentile(processing, 0.95),
        "physical_compute_mean_ms": (
            statistics.fmean(physical_compute) if physical_compute else 0.0
        ),
        "physical_queue_mean_ms": (
            statistics.fmean(physical_queue) if physical_queue else 0.0
        ),
        "scaled_processing_mean_ms": (
            statistics.fmean(scaled_processing) if scaled_processing else 0.0
        ),
        "total_delay_mean_ms": (
            statistics.fmean(total_delay) if total_delay else 0.0
        ),
        "delay_decomposition_max_error_ms": maximum_decomposition_error,
        "platform_overhead_mean_ms": statistics.fmean(overhead) if overhead else 0.0,
        "platform_overhead_p95_ms": _percentile(overhead, 0.95),
    }


def _pod_metrics(path: Path) -> dict:
    if not path.exists():
        return {
            "pod_samples": 0,
            "max_pods": 0,
            "max_ready_pods": 0,
            "pod_observation_s": 0.0,
            "pod_sample_errors": 0,
        }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = [
        datetime.fromisoformat(row["timestamp_utc"])
        for row in rows
        if row.get("timestamp_utc")
    ]
    return {
        "pod_samples": len(rows),
        "max_pods": max(
            (int(row["pod_count"]) for row in rows if row.get("pod_count")),
            default=0,
        ),
        "max_ready_pods": max(
            (int(row["ready_pod_count"]) for row in rows if row.get("ready_pod_count")),
            default=0,
        ),
        "pod_observation_s": (
            (timestamps[-1] - timestamps[0]).total_seconds()
            if len(timestamps) >= 2
            else 0.0
        ),
        "pod_sample_errors": sum(bool(row.get("error")) for row in rows),
    }


def _task_rows(run_dir: Path):
    source = run_dir / "tasks.csv"
    archive = run_dir / "tasks.csv.gz"
    if archive.exists():
        handle = gzip.open(archive, "rt", encoding="utf-8", newline="")
    elif source.exists():
        handle = source.open("r", encoding="utf-8", newline="")
    else:
        raise FileNotFoundError(f"task records not found in {run_dir}")
    with handle:
        yield from csv.DictReader(handle)


def _append_float(values: array, value: str | None) -> None:
    if value:
        values.append(float(value))


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _integer(value: str | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _compress_task_records(run_dir: Path) -> Path:
    source = run_dir / "tasks.csv"
    archive = run_dir / "tasks.csv.gz"
    if not source.exists():
        if archive.exists():
            return archive
        raise FileNotFoundError(f"task records not found in {run_dir}")
    temporary = run_dir / "tasks.csv.gz.tmp"
    with source.open("rb") as source_handle, gzip.open(
        temporary, "wb", compresslevel=6
    ) as archive_handle:
        shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)
    temporary.replace(archive)
    source.unlink()
    return archive


def _completed_run(run_dir: Path) -> bool:
    return (
        (run_dir / "summary.json").is_file()
        and (run_dir / "timing.json").is_file()
        and (
            (run_dir / "tasks.csv.gz").is_file()
            or (run_dir / "tasks.csv").is_file()
        )
    )


def _signature(
    config_path: Path,
    base_path: Path,
    checkpoints: tuple[Path, ...],
    endpoint: str,
    cases: tuple[ValidationCase, ...] = (),
    modes: tuple[str, ...] = (),
) -> str:
    digest = sha256()
    for path in (config_path, base_path, *checkpoints):
        digest.update(path.read_bytes())
    digest.update(endpoint.encode("utf-8"))
    digest.update(
        json.dumps(
            {
                "cases": [
                    {
                        "vehicle_count": case.vehicle_count,
                        "steps": case.steps,
                        "seed": case.seed,
                        "request_budget": case.request_budget,
                        "checkpoint": str(case.checkpoint.resolve()),
                    }
                    for case in cases
                ],
                "modes": modes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _estimated_storage_gb(
    task_counts: list[int],
    modes: tuple[str, ...],
    task_record_sample_rate: float,
) -> float:
    total_tasks = sum(task_counts)
    sampled_task_bytes = (
        total_tasks
        * len(modes)
        * task_record_sample_rate
        * BYTES_PER_RAW_TASK_RECORD
    )
    trace_bytes = (
        total_tasks * BYTES_PER_RAW_DECISION_TRACE
        if "knative_replay" in modes
        else 0
    )
    return (sampled_task_bytes + trace_bytes) / 1024**3


def _validation_cases(
    repo: Path,
    settings: dict,
    vehicle_counts: tuple[int, ...],
    steps: tuple[int, ...],
    checkpoint_override: Path | None,
    seeds: tuple[int, ...] | None = None,
) -> tuple[ValidationCase, ...]:
    if checkpoint_override is not None and len(vehicle_counts) != 1:
        raise ValueError("--checkpoint can only override one selected vehicle count")
    checkpoint_map = settings.get("checkpoints", {})
    request_budget_map = settings.get("request_budgets", {})
    cases = []
    for vehicle_count in vehicle_counts:
        key = str(vehicle_count)
        selected_seeds = seeds or tuple(
            int(value)
            for value in settings.get("seeds", [settings.get("seed", 0)])
        )
        checkpoint = None
        if checkpoint_override is not None:
            checkpoint = _resolve(repo, checkpoint_override)
        elif key in checkpoint_map:
            checkpoint = _resolve(repo, checkpoint_map[key])
        elif "checkpoint" in settings:
            checkpoint = _resolve(repo, settings["checkpoint"])
        elif not all(
            f"{vehicle_count}:{seed}" in checkpoint_map
            for seed in selected_seeds
        ):
            raise ValueError(f"no checkpoint configured for {vehicle_count} vehicles")
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(
                f"Frozen policy checkpoint not found: {checkpoint}. "
                "Pass -Checkpoint to the PowerShell runner."
            )
        if key in request_budget_map:
            request_budget = int(request_budget_map[key])
        elif "request_budget" in settings:
            request_budget = int(settings["request_budget"])
        else:
            raise ValueError(
                f"no Knative request budget configured for {vehicle_count} vehicles"
            )
        if request_budget <= 0:
            raise ValueError("Knative request budgets must be positive")
        for case_steps in steps:
            for seed in selected_seeds:
                seed_checkpoint_key = f"{vehicle_count}:{seed}"
                selected_checkpoint = (
                    _resolve(repo, checkpoint_map[seed_checkpoint_key])
                    if checkpoint_override is None
                    and seed_checkpoint_key in checkpoint_map
                    else checkpoint
                )
                if selected_checkpoint is None:
                    raise ValueError(
                        f"no checkpoint configured for {vehicle_count} vehicles "
                        f"and seed {seed}"
                    )
                if not selected_checkpoint.is_file():
                    raise FileNotFoundError(
                        "Frozen policy checkpoint not found: "
                        f"{selected_checkpoint}"
                    )
                cases.append(
                    ValidationCase(
                        vehicle_count=vehicle_count,
                        steps=case_steps,
                        seed=seed,
                        checkpoint=selected_checkpoint,
                        request_budget=request_budget,
                    )
                )
    return tuple(cases)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
