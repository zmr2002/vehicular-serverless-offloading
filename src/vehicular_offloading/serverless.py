from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
from threading import Lock, local
from time import perf_counter, sleep
from typing import Protocol

import requests

from .config import ServerlessConfig
from .domain import Task


SERVERLESS_DELAY_MODEL = "physical_compute_queue_plus_platform_v1"


def physical_cloud_delay_ms(
    task: Task,
    queue_length: int,
    cloud_compute_hz: float,
    queue_delay_fn,
) -> tuple[float, float]:
    return (
        task.compute_cycles / cloud_compute_hz * 1_000.0,
        queue_delay_fn(queue_length) * 1_000.0,
    )


def composed_service_delay_s(
    physical_compute_ms: float,
    physical_queue_ms: float,
    dispatch_queue_ms: float,
    platform_overhead_ms: float,
) -> float:
    return (
        physical_compute_ms
        + physical_queue_ms
        + dispatch_queue_ms
        + platform_overhead_ms
    ) / 1_000.0


@dataclass(slots=True, frozen=True)
class ServerlessMeasurement:
    service_delay_s: float
    processing_ms: float
    client_latency_ms: float | None
    platform_overhead_ms: float | None
    cold_start: bool
    instance_id: str
    checksum: str
    dispatch_queue_ms: float = 0.0
    http_latency_ms: float | None = None
    http_attempts: int = 0
    http_retry_count: int = 0
    retry_backoff_ms: float = 0.0
    physical_compute_ms: float = 0.0
    physical_queue_ms: float = 0.0
    scaled_processing_ms: float | None = None


class ServerlessBackend(Protocol):
    def execute(self, task: Task, queue_length: int, step: int) -> ServerlessMeasurement: ...


class AnalyticalServerlessBackend:
    def __init__(
        self,
        cloud_compute_hz: float,
        cold_start_s: float,
        queue_delay_fn,
        idle_steps_to_zero: int,
    ):
        self.cloud_compute_hz = cloud_compute_hz
        self.cold_start_s = cold_start_s
        self.queue_delay_fn = queue_delay_fn
        self.idle_steps_to_zero = idle_steps_to_zero
        self._last_step: int | None = None

    def execute(self, task: Task, queue_length: int, step: int) -> ServerlessMeasurement:
        cold = self.will_cold_start(step)
        self._last_step = step
        physical_compute_ms, physical_queue_ms = physical_cloud_delay_ms(
            task,
            queue_length,
            self.cloud_compute_hz,
            self.queue_delay_fn,
        )
        platform_s = self.cold_start_s if cold else 0.0
        service_delay = composed_service_delay_s(
            physical_compute_ms,
            physical_queue_ms,
            0.0,
            platform_s * 1_000.0,
        )
        checksum = sha256(f"{task.task_id}:{task.compute_cycles}".encode()).hexdigest()[:16]
        return ServerlessMeasurement(
            service_delay_s=service_delay,
            processing_ms=physical_compute_ms,
            client_latency_ms=None,
            platform_overhead_ms=platform_s * 1_000.0,
            cold_start=cold,
            instance_id="analytical-cloud",
            checksum=checksum,
            dispatch_queue_ms=0.0,
            http_latency_ms=None,
            http_attempts=0,
            http_retry_count=0,
            retry_backoff_ms=0.0,
            physical_compute_ms=physical_compute_ms,
            physical_queue_ms=physical_queue_ms,
            scaled_processing_ms=None,
        )

    def will_cold_start(self, step: int) -> bool:
        return self._last_step is None or step - self._last_step >= self.idle_steps_to_zero


class HttpKnativeBackend:
    def __init__(
        self,
        config: ServerlessConfig,
        cloud_compute_hz: float = 50e9,
        queue_delay_fn=lambda _queue: 0.0,
    ):
        self.config = config
        self.cloud_compute_hz = cloud_compute_hz
        self.queue_delay_fn = queue_delay_fn
        self._session_local = local()
        self._sessions: list[requests.Session] = []
        self._session_lock = Lock()
        self._observation_lock = Lock()
        self._request_lock = Lock()
        self._requests_started = 0
        self._warm_overhead_ema_s: float | None = None
        self._cold_overhead_ema_s: float | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=config.client_concurrency,
            thread_name_prefix="knative-client",
        )

    def submit(self, task: Task, queue_length: int, step: int) -> Future[ServerlessMeasurement]:
        submitted_at = perf_counter()
        return self._executor.submit(
            self._execute_reserved,
            task,
            queue_length,
            step,
            submitted_at,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._session_lock:
            sessions = tuple(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            self._session_local.session = session
            with self._session_lock:
                self._sessions.append(session)
        return session

    def execute(self, task: Task, queue_length: int, step: int) -> ServerlessMeasurement:
        return self._execute_reserved(
            task,
            queue_length,
            step,
            perf_counter(),
        )

    def _reserve_request(self) -> None:
        with self._request_lock:
            if (
                self.config.max_requests_per_run > 0
                and self._requests_started >= self.config.max_requests_per_run
            ):
                raise RuntimeError(
                    "Knative request budget exhausted: "
                    f"{self.config.max_requests_per_run} requests"
                )
            self._requests_started += 1

    def _execute_reserved(
        self,
        task: Task,
        queue_length: int,
        step: int,
        submitted_at: float,
    ) -> ServerlessMeasurement:
        worker_started = perf_counter()
        dispatch_queue_ms = max(0.0, (worker_started - submitted_at) * 1_000.0)
        work_units = max(1, min(self.config.max_work_units, round(task.compute_cycles / 1e6)))
        payload = {
            "task_id": task.task_id,
            "compute_cycles": task.compute_cycles,
            "data_size_mb": task.data_size_mb,
            "deadline_ms": task.deadline_s * 1_000.0,
            "work_units": work_units,
            "queue_length": queue_length,
            "simulation_step": step,
        }
        http_started = perf_counter()
        attempts = 0
        retry_backoff_s = 0.0
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        while True:
            self._reserve_request()
            attempts += 1
            try:
                response = self._session().post(
                    f"{self.config.endpoint.rstrip('/')}/v1/tasks",
                    json=payload,
                    timeout=self.config.timeout_s,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempts > self.config.max_retries:
                    raise
                delay_s = self._retry_delay_s(task.task_id, attempts)
                retry_backoff_s += delay_s
                sleep(delay_s)
                continue
            if (
                response.status_code in retryable_statuses
                and attempts <= self.config.max_retries
            ):
                response.close()
                delay_s = self._retry_delay_s(task.task_id, attempts)
                retry_backoff_s += delay_s
                sleep(delay_s)
                continue
            break
        http_latency_ms = (perf_counter() - http_started) * 1_000.0
        client_latency_ms = dispatch_queue_ms + http_latency_ms
        response.raise_for_status()
        body = response.json()
        processing_ms = float(body["processing_ms"])
        cold_start = bool(body["cold_start"])
        platform_overhead_ms = max(0.0, http_latency_ms - processing_ms)
        total_nonprocessing_ms = dispatch_queue_ms + platform_overhead_ms
        self._observe_overhead(total_nonprocessing_ms / 1_000.0, cold_start)
        physical_compute_ms, physical_queue_ms = physical_cloud_delay_ms(
            task,
            queue_length,
            self.cloud_compute_hz,
            self.queue_delay_fn,
        )
        service_delay_s = composed_service_delay_s(
            physical_compute_ms,
            physical_queue_ms,
            dispatch_queue_ms,
            platform_overhead_ms,
        )
        return ServerlessMeasurement(
            service_delay_s=service_delay_s,
            processing_ms=processing_ms,
            client_latency_ms=client_latency_ms,
            platform_overhead_ms=platform_overhead_ms,
            cold_start=cold_start,
            instance_id=str(body["instance_id"]),
            checksum=str(body["checksum"]),
            dispatch_queue_ms=dispatch_queue_ms,
            http_latency_ms=http_latency_ms,
            http_attempts=attempts,
            http_retry_count=attempts - 1,
            retry_backoff_ms=retry_backoff_s * 1_000.0,
            physical_compute_ms=physical_compute_ms,
            physical_queue_ms=physical_queue_ms,
            scaled_processing_ms=processing_ms,
        )

    def _retry_delay_s(self, task_id: str, attempt: int) -> float:
        if self.config.retry_backoff_s == 0:
            return 0.0
        digest = sha256(f"{task_id}:{attempt}".encode()).digest()
        jitter = 0.75 + 0.5 * digest[0] / 255.0
        return self.config.retry_backoff_s * (2 ** (attempt - 1)) * jitter

    def predicted_platform_overhead_s(self) -> float:
        """Return a warm-path estimate without inventing a cold-start penalty."""
        with self._observation_lock:
            if self._warm_overhead_ema_s is not None:
                return self._warm_overhead_ema_s
            return 0.0

    def _observe_overhead(self, overhead_s: float, cold_start: bool) -> None:
        with self._observation_lock:
            attribute = "_cold_overhead_ema_s" if cold_start else "_warm_overhead_ema_s"
            previous = getattr(self, attribute)
            updated = overhead_s if previous is None else 0.2 * overhead_s + 0.8 * previous
            setattr(self, attribute, updated)

    def health(self) -> dict:
        response = self._session().get(
            f"{self.config.endpoint.rstrip('/')}/healthz",
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        return response.json()
