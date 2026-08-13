from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vehicular_offloading.experiments import benchmark_serverless
from vehicular_offloading.serverless import ServerlessMeasurement


class _FakeBackend:
    calls: list[str] = []
    closed = False

    def __init__(self, _config):
        type(self).calls = []
        type(self).closed = False

    def execute(self, task, _queue_length, _step):
        type(self).calls.append(task.task_id)
        cold = len(type(self).calls) == 1
        return ServerlessMeasurement(
            service_delay_s=0.01,
            processing_ms=2.0,
            client_latency_ms=10.0 if cold else 3.0,
            platform_overhead_ms=8.0 if cold else 1.0,
            cold_start=cold,
            instance_id="fake",
            checksum="abc",
        )

    def health(self):
        type(self).calls.append("health")
        return {"status": "ok"}

    def close(self):
        type(self).closed = True


class ExperimentTests(unittest.TestCase):
    def test_cold_request_precedes_health_probe_and_bursts(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "vehicular_offloading.experiments.HttpKnativeBackend", _FakeBackend
        ):
            output = benchmark_serverless(
                "http://example.invalid", temp, concurrencies=(1,), requests_per_level=1
            )
            with Path(output).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["phase"] for row in rows], ["cold", "warm", "burst"])
            self.assertTrue(_FakeBackend.calls[0].startswith("bench-cold"))
            self.assertEqual(_FakeBackend.calls[2], "health")
            self.assertTrue(_FakeBackend.closed)


if __name__ == "__main__":
    unittest.main()
