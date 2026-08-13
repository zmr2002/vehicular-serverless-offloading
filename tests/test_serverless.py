from __future__ import annotations

import unittest
from time import sleep

import requests

from vehicular_offloading.config import ServerlessConfig
from vehicular_offloading.domain import Task
from vehicular_offloading.serverless import (
    AnalyticalServerlessBackend,
    HttpKnativeBackend,
    composed_service_delay_s,
    physical_cloud_delay_ms,
)


class _Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {
            "processing_ms": 1.0,
            "cold_start": False,
            "instance_id": "test-instance",
            "checksum": "0123456789abcdef",
        }

    def close(self) -> None:
        return None


class _Session:
    def post(self, *_args, **_kwargs) -> _Response:
        return _Response()

    def close(self) -> None:
        return None


class _SlowSession(_Session):
    def post(self, *_args, **_kwargs) -> _Response:
        sleep(0.03)
        return _Response()


class _SequenceSession(_Session):
    def __init__(self, statuses: list[int]):
        self.statuses = statuses
        self.calls = 0

    def post(self, *_args, **_kwargs) -> _Response:
        status = self.statuses[self.calls]
        self.calls += 1
        return _Response(status)


class ServerlessBackendTests(unittest.TestCase):
    def test_physical_delay_is_backend_independent(self) -> None:
        task = Task(
            task_id="task",
            vehicle_id="vehicle",
            compute_cycles=5e9,
            data_size_mb=1.0,
            deadline_s=2.0,
            urgency=0.0,
            created_step=0,
        )
        physical_compute_ms, physical_queue_ms = physical_cloud_delay_ms(
            task,
            7,
            50e9,
            lambda queue: queue / 100.0,
        )
        self.assertAlmostEqual(physical_compute_ms, 100.0)
        self.assertAlmostEqual(physical_queue_ms, 70.0)
        self.assertAlmostEqual(
            composed_service_delay_s(
                physical_compute_ms,
                physical_queue_ms,
                dispatch_queue_ms=0.0,
                platform_overhead_ms=0.0,
            ),
            0.17,
        )

    def test_analytical_backend_decomposes_physical_and_platform_delay(self) -> None:
        backend = AnalyticalServerlessBackend(
            cloud_compute_hz=50e9,
            cold_start_s=0.1,
            queue_delay_fn=lambda _queue: 0.2,
            idle_steps_to_zero=50,
        )
        task = Task(
            task_id="task",
            vehicle_id="vehicle",
            compute_cycles=5e9,
            data_size_mb=1.0,
            deadline_s=2.0,
            urgency=0.0,
            created_step=0,
        )
        measured = backend.execute(task, 4, 0)
        self.assertAlmostEqual(measured.physical_compute_ms, 100.0)
        self.assertAlmostEqual(measured.physical_queue_ms, 200.0)
        self.assertAlmostEqual(measured.platform_overhead_ms, 100.0)
        self.assertAlmostEqual(measured.service_delay_s, 0.4)
        self.assertIsNone(measured.scaled_processing_ms)

    def test_request_budget_is_enforced(self) -> None:
        config = ServerlessConfig(max_requests_per_run=1)
        backend = HttpKnativeBackend(config)
        backend._session = lambda: _Session()
        task = Task(
            task_id="task",
            vehicle_id="vehicle",
            compute_cycles=1e9,
            data_size_mb=1.0,
            deadline_s=2.0,
            urgency=0.0,
            created_step=0,
        )
        try:
            backend.execute(task, 0, 0)
            with self.assertRaisesRegex(RuntimeError, "request budget exhausted"):
                backend.execute(task, 0, 0)
        finally:
            backend.close()

    def test_thread_pool_wait_is_included_in_service_delay(self) -> None:
        config = ServerlessConfig(client_concurrency=1)
        backend = HttpKnativeBackend(
            config,
            cloud_compute_hz=50e9,
            queue_delay_fn=lambda _queue: 0.2,
        )
        backend._session = lambda: _SlowSession()
        task = Task(
            task_id="task",
            vehicle_id="vehicle",
            compute_cycles=1e9,
            data_size_mb=1.0,
            deadline_s=2.0,
            urgency=0.0,
            created_step=0,
        )
        try:
            first = backend.submit(task, 0, 0)
            second = backend.submit(task, 0, 0)
            first.result()
            measured = second.result()
            self.assertGreaterEqual(measured.dispatch_queue_ms, 20.0)
            self.assertAlmostEqual(
                measured.client_latency_ms,
                measured.dispatch_queue_ms + measured.http_latency_ms,
                places=6,
            )
            self.assertAlmostEqual(
                measured.service_delay_s * 1_000.0,
                measured.physical_compute_ms
                + measured.physical_queue_ms
                + measured.dispatch_queue_ms
                + measured.platform_overhead_ms,
                places=6,
            )
            self.assertAlmostEqual(measured.physical_compute_ms, 20.0)
            self.assertAlmostEqual(measured.physical_queue_ms, 200.0)
            self.assertEqual(measured.scaled_processing_ms, measured.processing_ms)
            self.assertAlmostEqual(
                measured.platform_overhead_ms,
                measured.http_latency_ms - measured.processing_ms,
                places=6,
            )
        finally:
            backend.close()

    def test_retryable_gateway_error_is_measured_and_budgeted(self) -> None:
        config = ServerlessConfig(
            max_requests_per_run=2,
            max_retries=1,
            retry_backoff_s=0.0,
        )
        backend = HttpKnativeBackend(config)
        session = _SequenceSession([502, 200])
        backend._session = lambda: session
        task = Task(
            task_id="retry-task",
            vehicle_id="vehicle",
            compute_cycles=1e9,
            data_size_mb=1.0,
            deadline_s=2.0,
            urgency=0.0,
            created_step=0,
        )
        try:
            measured = backend.execute(task, 0, 0)
            self.assertEqual(measured.http_attempts, 2)
            self.assertEqual(measured.http_retry_count, 1)
            self.assertEqual(measured.retry_backoff_ms, 0.0)
            self.assertEqual(backend._requests_started, 2)
            with self.assertRaisesRegex(RuntimeError, "request budget exhausted"):
                backend.execute(task, 0, 0)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
