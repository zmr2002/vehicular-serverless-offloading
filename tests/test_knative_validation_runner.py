from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run-knative-validation.py"
    spec = importlib.util.spec_from_file_location("knative_validation_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


class KnativeValidationRunnerTests(unittest.TestCase):
    def test_storage_estimate_applies_sampling_and_one_compact_trace(self) -> None:
        estimated = RUNNER._estimated_storage_gb(
            [1_000_000],
            ("analytical", "knative_replay", "knative_closed_loop"),
            0.001,
        )
        expected_bytes = (
            1_000_000 * 3 * 0.001 * RUNNER.BYTES_PER_RAW_TASK_RECORD
            + 1_000_000 * RUNNER.BYTES_PER_RAW_DECISION_TRACE
        )
        self.assertAlmostEqual(estimated, expected_bytes / 1024**3)

    def test_vehicle_specific_checkpoints_and_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            checkpoints = {}
            for vehicles in (1000, 2000):
                path = repo / f"{vehicles}.pt"
                path.write_bytes(b"checkpoint")
                checkpoints[str(vehicles)] = path.name
            settings = {
                "checkpoints": checkpoints,
                "request_budgets": {"1000": 10, "2000": 20},
            }
            cases = RUNNER._validation_cases(
                repo,
                settings,
                (1000, 2000),
                (2000,),
                None,
            )
            self.assertEqual([case.vehicle_count for case in cases], [1000, 2000])
            self.assertEqual([case.request_budget for case in cases], [10, 20])

    def test_seed_specific_checkpoints_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            checkpoints = {}
            for seed in (61, 62):
                path = repo / f"1000-{seed}.pt"
                path.write_bytes(str(seed).encode("ascii"))
                checkpoints[f"1000:{seed}"] = path.name
            cases = RUNNER._validation_cases(
                repo,
                {
                    "checkpoints": checkpoints,
                    "request_budgets": {"1000": 10},
                },
                (1000,),
                (2000,),
                None,
                (61, 62),
            )
            self.assertEqual([case.seed for case in cases], [61, 62])
            self.assertEqual(
                [case.checkpoint.name for case in cases],
                ["1000-61.pt", "1000-62.pt"],
            )

    def test_streaming_comparison_includes_dispatch_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analytical = root / "analytical"
            live = root / "live"
            analytical.mkdir()
            live.mkdir()
            for run_dir in (analytical, live):
                RUNNER._write_json_atomic(
                    run_dir / "environment.json",
                    {
                        "serverless_delay_model": (
                            RUNNER.SERVERLESS_DELAY_MODEL
                        )
                    },
                )
            fields = [
                "task_id",
                "action",
                "success",
                "dispatch_queue_ms",
                "http_latency_ms",
                "client_latency_ms",
                "processing_ms",
                "platform_overhead_ms",
                "preprocessing_delay_ms",
                "radio_delay_ms",
                "physical_compute_ms",
                "physical_queue_ms",
                "scaled_processing_ms",
                "total_delay_ms",
                "http_attempts",
                "http_retry_count",
                "retry_backoff_ms",
                "cold_start",
                "instance_id",
            ]
            self._write(
                analytical / "tasks.csv",
                fields,
                [
                    {"task_id": "a", "action": "local", "success": "1"},
                    {"task_id": "b", "action": "v2i", "success": "1"},
                ],
            )
            self._write(
                live / "tasks.csv",
                fields,
                [
                    {
                        "task_id": "a",
                        "action": "v2i",
                        "success": "0",
                        "dispatch_queue_ms": "10",
                        "http_latency_ms": "20",
                        "client_latency_ms": "30",
                        "processing_ms": "5",
                        "platform_overhead_ms": "15",
                        "preprocessing_delay_ms": "1",
                        "radio_delay_ms": "2",
                        "physical_compute_ms": "20",
                        "physical_queue_ms": "30",
                        "scaled_processing_ms": "5",
                        "total_delay_ms": "78",
                        "http_attempts": "2",
                        "http_retry_count": "1",
                        "retry_backoff_ms": "5",
                        "cold_start": "1",
                        "instance_id": "one",
                    },
                    {
                        "task_id": "b",
                        "action": "v2i",
                        "success": "1",
                        "dispatch_queue_ms": "40",
                        "http_latency_ms": "60",
                        "client_latency_ms": "100",
                        "processing_ms": "10",
                        "platform_overhead_ms": "50",
                        "preprocessing_delay_ms": "1",
                        "radio_delay_ms": "2",
                        "physical_compute_ms": "20",
                        "physical_queue_ms": "30",
                        "scaled_processing_ms": "10",
                        "total_delay_ms": "143",
                        "http_attempts": "1",
                        "http_retry_count": "0",
                        "retry_backoff_ms": "0",
                        "cold_start": "0",
                        "instance_id": "two",
                    },
                ],
            )
            compared = RUNNER._compare_runs(analytical, live)
            self.assertEqual(compared["matched_task_records"], 2)
            self.assertEqual(compared["live_v2i_requests"], 2)
            self.assertEqual(compared["live_v2i_failures"], 1)
            self.assertEqual(compared["cold_start_flags"], 1)
            self.assertEqual(compared["retried_requests"], 1)
            self.assertEqual(compared["total_http_retries"], 1)
            self.assertEqual(compared["distinct_instances"], 2)
            self.assertAlmostEqual(compared["action_change_rate"], 0.5)
            self.assertAlmostEqual(compared["dispatch_queue_mean_ms"], 25.0)
            self.assertAlmostEqual(compared["http_latency_mean_ms"], 40.0)
            self.assertAlmostEqual(compared["physical_compute_mean_ms"], 20.0)
            self.assertAlmostEqual(compared["scaled_processing_mean_ms"], 7.5)
            self.assertAlmostEqual(
                compared["delay_decomposition_max_error_ms"],
                0.0,
            )

    def test_full_run_serverless_metrics_override_sampled_counts(self) -> None:
        sampled = {
            "live_v2i_requests": 1,
            "live_v2i_failures": 0,
        }
        aggregate = RUNNER._aggregate_serverless_metrics(
            {
                "serverless_http_request_count": 397,
                "serverless_v2i_failure_count": 1,
                "serverless_retried_request_count": 2,
                "serverless_http_retry_count": 3,
                "serverless_cold_start_count": 1,
                "serverless_distinct_instance_count": 4,
                "avg_serverless_client_latency_ms": 20.0,
                "p95_serverless_client_latency_ms": 30.0,
                "max_serverless_client_latency_ms": 40.0,
                "max_serverless_cold_client_latency_ms": 40.0,
                "p95_serverless_warm_client_latency_ms": 29.0,
                "p95_serverless_dispatch_queue_ms": 2.0,
                "p95_serverless_http_latency_ms": 28.0,
                "p95_serverless_platform_overhead_ms": 25.0,
                "avg_serverless_physical_compute_ms": 50.0,
                "avg_serverless_scaled_processing_ms": 1.0,
                "serverless_delay_decomposition_max_error_ms": 0.0,
            },
            sampled,
        )
        self.assertEqual(aggregate["live_v2i_requests"], 397)
        self.assertEqual(aggregate["live_v2i_failures"], 1)
        self.assertEqual(aggregate["distinct_instances"], 4)

    @staticmethod
    def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
