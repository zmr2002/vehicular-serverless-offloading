from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from threading import Lock
from time import sleep
import unittest
from unittest.mock import patch

import numpy as np

from vehicular_offloading.config import DecisionConfig, DQNConfig, ServerlessConfig, SimulationConfig
from vehicular_offloading.domain import OffloadAction, OffloadEstimate, Task, VehicleState
from vehicular_offloading.mobility import SyntheticMobilityProvider, TraceCachingMobilityProvider
from vehicular_offloading.serverless import ServerlessMeasurement
from vehicular_offloading.simulation import SimulationRunner


class SimulationTests(unittest.TestCase):
    def test_batch_cloud_capacity_counts_full_synchronous_demand(self):
        config = SimulationConfig()
        runner = SimulationRunner(config)
        try:
            ratio = runner._batch_cloud_capacity_ratio(
                current_request_count=20,
                anticipated_request_count=80.0,
                anticipated_cycles=1.0e9,
            )
        finally:
            runner.mobility.close()
        self.assertEqual(ratio, 1.0)

    def _config(self, output: str) -> SimulationConfig:
        return SimulationConfig(
            steps=3,
            vehicle_count=8,
            seed=123,
            task_probability=0.5,
            strategy="hybrid_stackelberg",
            mobility="synthetic",
            output_dir=output,
            service_role_mode="dynamic_idle",
            dqn=DQNConfig(
                replay_capacity=100,
                batch_size=4,
                warmup_transitions=4,
                target_update_interval=2,
            ),
        )

    def test_fixed_seed_is_reproducible_and_logs_have_one_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            first, first_dir = SimulationRunner(self._config(temp)).run()
            second, second_dir = SimulationRunner(self._config(temp)).run()
            self.assertEqual(asdict(first), asdict(second))
            ratios = first.local_offload_ratio + first.v2v_offload_ratio + first.v2i_offload_ratio
            self.assertAlmostEqual(ratios, 1.0)
            self.assertEqual(first.completed_steps, 3)
            self.assertEqual(first.configured_vehicle_count, 8)
            header = (Path(first_dir) / "tasks.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("avg_success_rate", header)
            self.assertEqual(first.replay_size, first.total_tasks)
            self.assertEqual(first.dqn_transitions, first.total_tasks)
            self.assertGreaterEqual(first.oracle_success_rate, first.success_rate)
            self.assertGreaterEqual(first.avg_server_distance_m, 0.0)
            self.assertGreaterEqual(first.avg_decision_regret_s, 0.0)
            self.assertGreaterEqual(first.mandatory_remote_task_ratio, 0.0)
            self.assertGreaterEqual(first.avg_mandatory_remote_cycles_per_step, 0.0)
            self.assertGreaterEqual(first.mandatory_remote_to_cloud_capacity_ratio, 0.0)
            self.assertAlmostEqual(
                first.task_vehicle_step_ratio + first.service_vehicle_step_ratio,
                1.0,
            )
            self.assertGreaterEqual(first.offered_vehicle_compute_load_ratio, 0.0)
            self.assertGreaterEqual(first.p95_source_workload_s, 0.0)
            self.assertGreaterEqual(first.reachable_v2v_task_ratio, first.v2v_rescuable_task_ratio)
            self.assertGreaterEqual(first.intrinsic_local_infeasible_task_ratio, 0.0)
            timing = json.loads((Path(first_dir) / "timing.json").read_text(encoding="utf-8"))
            self.assertGreater(timing["wall_clock_s"], 0.0)
            self.assertEqual(len(timing["segments"]), 1)
            self.assertEqual(timing["segments"][0]["tasks"], first.total_tasks)
            self.assertIn("topology_and_pricing", timing["phase_seconds"])
            self.assertTrue((Path(first_dir) / "dqn-policy.pt").exists())

    def test_task_records_can_be_suppressed_for_screening_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.record_task_records = False
            summary, run_dir = SimulationRunner(config).run()
            rows = (Path(run_dir) / "tasks.csv").read_text(encoding="utf-8").splitlines()
            self.assertGreater(summary.total_tasks, 0)
            self.assertEqual(len(rows), 1)
            recording = json.loads(
                (Path(run_dir) / "task-recording.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recording["records_seen"], summary.total_tasks)
            self.assertEqual(recording["records_written"], 0)

    def test_task_records_can_be_deterministically_sampled(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.task_record_sample_rate = 0.5
            first, first_dir = SimulationRunner(config).run()
            second, second_dir = SimulationRunner(config).run()
            first_rows = (Path(first_dir) / "tasks.csv").read_text(encoding="utf-8")
            second_rows = (Path(second_dir) / "tasks.csv").read_text(encoding="utf-8")
            self.assertEqual(first_rows, second_rows)
            recording = json.loads(
                (Path(first_dir) / "task-recording.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recording["records_seen"], first.total_tasks)
            self.assertGreater(recording["records_written"], 0)
            self.assertLess(recording["records_written"], first.total_tasks)

    def test_decision_trace_replay_freezes_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "actions.jsonl.gz"
            record = self._config(str(Path(temp) / "record"))
            record.decision_trace_mode = "record"
            record.decision_trace_path = str(trace)
            recorded, recorded_dir = SimulationRunner(record).run()
            replay = self._config(str(Path(temp) / "replay"))
            replay.decision_trace_mode = "replay"
            replay.decision_trace_path = str(trace)
            replayed, replayed_dir = SimulationRunner(replay).run()
            self.assertTrue(trace.exists())
            self.assertEqual(asdict(recorded), asdict(replayed))
            self.assertEqual(
                (Path(recorded_dir) / "tasks.csv").read_text(encoding="utf-8"),
                (Path(replayed_dir) / "tasks.csv").read_text(encoding="utf-8"),
            )

    def test_follower_pricing_is_hypothetical_and_records_one_broadcast_per_step(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.cloud_pricing_mode = "follower_best_response"
            config.cloud_price_candidate_count = 3
            config.decision.hybrid_fusion_mode = "confidence_gated"
            summary, run_dir = SimulationRunner(config).run()
            pricing = [
                json.loads(line)
                for line in (Path(run_dir) / "pricing.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(pricing), summary.completed_steps)
            self.assertTrue(
                all(
                    config.cloud_min_price
                    <= row["selected_price"]
                    <= config.cloud_max_price
                    for row in pricing
                )
            )
            self.assertTrue(
                all(row["predicted_cloud_share"] is not None for row in pricing)
            )
            self.assertTrue(
                all(row["realized_cloud_share"] is not None for row in pricing)
            )
            self.assertTrue(
                all(row["prediction_error"] is not None for row in pricing)
            )
            self.assertTrue(
                all(row["outer_iterations"] == 2 for row in pricing)
            )
            self.assertEqual(summary.dqn_transitions, summary.total_tasks)

    def test_batched_candidate_pricing_matches_sequential_pricing(self):
        with tempfile.TemporaryDirectory() as temp:
            sequential = self._config(str(Path(temp) / "sequential"))
            sequential.cloud_pricing_mode = "follower_best_response"
            sequential.cloud_price_candidate_count = 3
            sequential.cloud_price_batch_candidates = False
            batched = self._config(str(Path(temp) / "batched"))
            batched.cloud_pricing_mode = "follower_best_response"
            batched.cloud_price_candidate_count = 3
            batched.cloud_price_batch_candidates = True
            sequential_summary, sequential_dir = SimulationRunner(sequential).run()
            batched_summary, batched_dir = SimulationRunner(batched).run()
            self.assertEqual(asdict(sequential_summary), asdict(batched_summary))
            self.assertEqual(
                (Path(sequential_dir) / "pricing.jsonl").read_text(encoding="utf-8"),
                (Path(batched_dir) / "pricing.jsonl").read_text(encoding="utf-8"),
            )

    def test_price_response_state_updates_queue_delays_and_payments(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            runner = SimulationRunner(config)
            try:
                snapshot = {
                    "states": np.zeros((2, 20), dtype=np.float32),
                    "deadlines": np.asarray([2.0, 4.0]),
                }
                delays = np.asarray(
                    [[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]],
                    dtype=np.float64,
                )
                payments = np.asarray(
                    [[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]],
                    dtype=np.float64,
                )
                states = runner._price_response_states(
                    snapshot,
                    price=0.5,
                    anticipated_queue=5,
                    anticipated_capacity_ratio=1.25,
                    delays=delays,
                    payments=payments,
                )
            finally:
                runner.mobility.close()
            self.assertTrue(np.allclose(states[:, 8], 0.5))
            self.assertTrue(np.allclose(states[:, 9], 0.5))
            self.assertTrue(np.allclose(states[:, 10], 1.25))
            self.assertTrue(
                np.allclose(
                    states[:, 11:14],
                    np.asarray([[0.5, 1.0, 1.5], [0.5, 1.0, 2.0]]),
                )
            )
            self.assertTrue(np.allclose(states[:, 17], [1.0, 2.0]))
            self.assertTrue(np.allclose(states[:, 18], [2.0, 2.0]))

    def test_stackelberg_price_response_matches_realized_synchronous_demand(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.strategy = "stackelberg"
            config.cloud_pricing_mode = "follower_best_response"
            config.cloud_price_candidate_count = 3
            _, run_dir = SimulationRunner(config).run()
            pricing = [
                json.loads(line)
                for line in (Path(run_dir) / "pricing.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                all(
                    abs(row["prediction_error"]) < 1e-12
                    for row in pricing
                )
            )

    def test_training_interval_survives_full_replay_buffer(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(
                steps=5,
                vehicle_count=8,
                seed=9,
                task_probability=1.0,
                strategy="dqn",
                mobility="synthetic",
                output_dir=temp,
                service_role_mode="dynamic_idle",
                dqn=DQNConfig(
                    replay_capacity=8,
                    batch_size=4,
                    warmup_transitions=4,
                    training_interval=4,
                    target_update_interval=2,
                ),
            )
            summary, _ = SimulationRunner(config).run()
            self.assertEqual(summary.dqn_transitions, 40)
            self.assertEqual(summary.replay_size, 8)
            self.assertEqual(summary.dqn_updates, 10)

    def test_dqn_transitions_link_successive_tasks_from_the_same_vehicle(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(
                steps=2,
                vehicle_count=4,
                seed=9,
                task_probability=1.0,
                strategy="dqn",
                mobility="synthetic",
                output_dir=temp,
                dqn=DQNConfig(
                    replay_capacity=20,
                    batch_size=4,
                    warmup_transitions=20,
                ),
            )
            runner = SimulationRunner(config)

            def deterministic_tasks(step, vehicles):
                return [
                    Task(
                        f"step-{step}-{vehicle_id}",
                        vehicle_id,
                        1e9 + (step * 100 + index) * 1e6,
                        10.0,
                        2.0,
                        0.5,
                        step,
                    )
                    for index, vehicle_id in enumerate(sorted(vehicles))
                ]

            runner._generate_tasks = deterministic_tasks
            summary, _ = runner.run()
            self.assertEqual(summary.dqn_transitions, 8)
            linked = runner.dqn.replay._items[:4]
            for state, _action, _reward, next_state, done, _mask in linked:
                self.assertFalse(done)
                self.assertAlmostEqual(float(next_state[0] - state[0]), 0.02, places=6)

    def test_analytical_queue_is_distributed_across_bounded_instances(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = SimulationRunner(self._config(temp))
            self.assertEqual(runner._scaled_cloud_queue(0), 0)
            self.assertEqual(runner._scaled_cloud_queue(10), 10)
            self.assertEqual(runner._scaled_cloud_queue(11), 6)
            self.assertEqual(runner._scaled_cloud_queue(100), 10)
            self.assertEqual(runner._scaled_cloud_queue(101), 11)

    def test_cloud_target_is_bounded_by_request_and_compute_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = SimulationRunner(self._config(temp))
            tasks = [
                Task(f"t-{index}", "v", 3e9, 1.0, 2.5, 0.0, 0)
                for index in range(1_000)
            ]
            # Request capacity is 10 instances * 10 requests * 85% reserve.
            self.assertAlmostEqual(runner._cloud_capacity_target(tasks), 0.085)
            self.assertAlmostEqual(
                runner._predicted_cloud_capacity_ratio(101, 3e9),
                1.01,
            )

    def test_uniform_compute_sampler_is_continuous_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = SimulationRunner(self._config(temp))
            samples = [runner._sample_compute_cycles() for _ in range(100)]
            self.assertTrue(all(1e9 <= value <= 5e9 for value in samples))
            self.assertTrue(any(value not in {1e9, 5e9} for value in samples))

    def test_uniform_deadline_sampler_is_continuous_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.task_probability = 1.0
            runner = SimulationRunner(config)
            vehicles = runner._vehicle_states(runner.mobility.step(0).vehicles)
            deadlines = [
                task.deadline_s
                for step in range(10)
                for task in runner._generate_tasks(step, vehicles)
            ]
            self.assertTrue(deadlines)
            self.assertTrue(all(2.0 <= value <= 3.0 for value in deadlines))
            self.assertTrue(any(value not in {2.0, 3.0} for value in deadlines))

    def test_fixed_service_roles_are_stable_and_do_not_generate_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(
                vehicle_count=50,
                task_probability=1.0,
                output_dir=temp,
                service_role_mode="fixed_ratio",
                service_vehicle_ratio=0.3,
            )
            runner = SimulationRunner(config)
            vehicles = runner._vehicle_states(runner.mobility.step(0).vehicles)
            service_ids = {
                vehicle_id
                for vehicle_id, vehicle in vehicles.items()
                if runner._is_fixed_service_vehicle(vehicle_id)
            }
            for vehicle_id, vehicle in vehicles.items():
                vehicle.is_service = vehicle_id in service_ids
            tasks = runner._generate_tasks(0, vehicles)
            self.assertTrue(service_ids)
            self.assertTrue(service_ids.isdisjoint(task.vehicle_id for task in tasks))
            self.assertEqual(
                service_ids,
                {
                    vehicle_id
                    for vehicle_id in vehicles
                    if runner._is_fixed_service_vehicle(vehicle_id)
                },
            )

    def test_service_workload_persists_and_drains_between_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = SimulationRunner(self._config(temp))
            runner.service_workload_cycles["service"] = 6e9
            runner.workload_compute_hz["service"] = runner.config.service_compute_hz
            runner._advance_workloads()
            self.assertEqual(runner.service_workload_cycles["service"], 4e9)
            runner._advance_workloads()
            self.assertEqual(runner.service_workload_cycles["service"], 2e9)
            runner._advance_workloads()
            self.assertNotIn("service", runner.service_workload_cycles)

    def test_remote_compute_energy_is_not_charged_to_source_battery(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = SimulationRunner(self._config(temp))
            task = Task("task", "vehicle", 1e9, 1.0, 2.0, 0.5, 0)
            vehicle = VehicleState("vehicle", (0.0, 0.0), 0.0, 2e9)
            estimate = OffloadEstimate(OffloadAction.V2I, 0.5, 10.0, 0.1)
            runner._apply_resource_effects(
                OffloadAction.V2I,
                estimate,
                task,
                vehicle,
                {vehicle.vehicle_id: vehicle},
                {},
                0.1,
            )
            cloud_energy = 1e9 / runner.config.cloud_compute_hz * 80.0
            expected_source_energy = estimate.energy_j - cloud_energy
            self.assertAlmostEqual(
                vehicle.energy_level,
                1.0 - expected_source_energy / runner.config.vehicle_battery_capacity_j,
            )

    def test_batch_execution_adds_queue_after_the_shared_decision_snapshot(self):
        target = VehicleState(
            "service",
            (0.0, 0.0),
            0.0,
            2e9,
            workload_cycles=2e9,
        )
        estimate = OffloadEstimate(
            OffloadAction.V2V,
            delay_s=1.5,
            energy_j=10.0,
            payment=0.1,
            target_id=target.vehicle_id,
        )
        decision = {
            "action": OffloadAction.V2V,
            "candidate_map": {OffloadAction.V2V: estimate},
            "v2v_target_workload_s": 0.0,
        }
        adjusted = SimulationRunner._batch_adjusted_estimate(
            decision,
            {target.vehicle_id: target},
        )
        self.assertAlmostEqual(adjusted.delay_s, 2.5)

    def test_batch_execution_replaces_forecast_with_realized_v2v_queue(self):
        target = VehicleState(
            "service",
            (0.0, 0.0),
            0.0,
            2e9,
            workload_cycles=1e9,
        )
        estimate = OffloadEstimate(
            OffloadAction.V2V,
            delay_s=2.0,
            energy_j=10.0,
            payment=0.1,
            target_id=target.vehicle_id,
        )
        decision = {
            "action": OffloadAction.V2V,
            "candidate_map": {OffloadAction.V2V: estimate},
            "v2v_target_workload_s": 0.0,
            "anticipated_v2v_queue_s": 1.0,
        }
        adjusted = SimulationRunner._batch_adjusted_estimate(
            decision,
            {target.vehicle_id: target},
        )
        self.assertAlmostEqual(adjusted.delay_s, 1.5)

    def test_expected_v2v_queue_uses_half_of_other_batch_work(self):
        target = VehicleState("service", (0.0, 0.0), 0.0, 2e9)
        snapshot = {
            "cycles": np.asarray([2e9, 4e9, 3e9], dtype=np.float64),
            "target_ids": ["service", "service", None],
            "target_indices": np.asarray([0, 0, -1], dtype=np.int32),
            "target_compute_hz": np.asarray([2e9, 2e9, 1.0]),
            "vehicles": {"service": target},
        }
        probabilities = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        queue_s = SimulationRunner._expected_v2v_batch_queue_s(
            snapshot,
            probabilities,
        )
        self.assertAlmostEqual(queue_s[0], 1.0)
        self.assertAlmostEqual(queue_s[1], 0.5)
        self.assertEqual(queue_s[2], 0.0)

    def test_all_tasks_decide_from_one_cloud_queue_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(
                steps=1,
                vehicle_count=20,
                seed=31,
                task_probability=1.0,
                strategy="greedy",
                mobility="synthetic",
                output_dir=temp,
                service_role_mode="fixed_ratio",
                service_vehicle_ratio=0.2,
            )
            runner = SimulationRunner(config)
            observed_cloud_queues: list[int] = []

            def choose_v2i(_strategy, context, *_args, **_kwargs):
                observed_cloud_queues.append(context.cloud_queue_length)
                return OffloadAction.V2I, None, False

            with patch("vehicular_offloading.simulation.choose_action", choose_v2i):
                summary, _ = runner.run()

            self.assertGreater(summary.total_tasks, 1)
            self.assertEqual(len(observed_cloud_queues), summary.total_tasks)
            self.assertEqual(len(set(observed_cloud_queues)), 1)
            self.assertEqual(summary.v2i_offload_ratio, 1.0)
            self.assertGreater(runner.cloud_workload_cycles, 0.0)

    def test_knative_requests_from_one_step_execute_concurrently(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(
                steps=1,
                vehicle_count=20,
                seed=5,
                task_probability=1.0,
                task_compute_choices=(5e9,),
                task_data_min_mb=1.0,
                task_data_max_mb=1.0,
                task_deadlines_s=(10.0,),
                strategy="greedy",
                backend="knative",
                mobility="synthetic",
                output_dir=temp,
                serverless=ServerlessConfig(client_concurrency=4),
            )
            runner = SimulationRunner(config)
            active = 0
            maximum_active = 0
            lock = Lock()

            def fake_execute(_task, _queue_length, _step, _submitted_at):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                sleep(0.01)
                with lock:
                    active -= 1
                return ServerlessMeasurement(
                    service_delay_s=0.01,
                    processing_ms=2.0,
                    client_latency_ms=10.0,
                    platform_overhead_ms=8.0,
                    cold_start=False,
                    instance_id="test-instance",
                    checksum="abc",
                )

            runner.backend._execute_reserved = fake_execute
            self.assertEqual(runner._predicted_platform_overhead_s(0), 0.0)
            summary, run_dir = runner.run()
            self.assertGreater(maximum_active, 1)
            self.assertEqual(summary.v2i_offload_ratio, 1.0)
            rows = (Path(run_dir) / "tasks.csv").read_text(encoding="utf-8")
            self.assertIn("dispatch_queue_ms", rows.splitlines()[0])
            self.assertIn("platform_overhead_ms", rows.splitlines()[0])
            self.assertIn("physical_compute_ms", rows.splitlines()[0])
            self.assertIn("physical_queue_ms", rows.splitlines()[0])
            self.assertIn("scaled_processing_ms", rows.splitlines()[0])
            self.assertIn("total_delay_ms", rows.splitlines()[0])
            self.assertIn("test-instance", rows)

    def test_mobility_trace_replay_is_exact_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(steps=4, vehicle_count=6, seed=17, output_dir=temp)
            cache_path = Path(temp) / "trace.jsonl.gz"
            recorder = TraceCachingMobilityProvider(
                config, SyntheticMobilityProvider(config), cache_path, "test-signature"
            )
            recorder.start()
            recorded = [recorder.step(step) for step in range(config.steps)]
            recorder.close()
            self.assertTrue(cache_path.exists())

            replay = TraceCachingMobilityProvider(
                config, SyntheticMobilityProvider(config), cache_path, "test-signature"
            )
            replay.start()
            replayed = [replay.step(step) for step in range(config.steps)]
            replay.close()
            self.assertEqual(recorded, replayed)

    def test_concurrent_mobility_writers_publish_one_trace_without_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SimulationConfig(steps=4, vehicle_count=6, seed=17, output_dir=temp)
            cache_path = Path(temp) / "trace.jsonl.gz"
            first = TraceCachingMobilityProvider(
                config, SyntheticMobilityProvider(config), cache_path, "shared-signature"
            )
            second = TraceCachingMobilityProvider(
                config, SyntheticMobilityProvider(config), cache_path, "shared-signature"
            )
            first.start()
            second.start()
            first_frames = [first.step(step) for step in range(config.steps)]
            second_frames = [second.step(step) for step in range(config.steps)]
            first.close()
            second.close()

            self.assertEqual(first_frames, second_frames)
            self.assertTrue(cache_path.exists())
            self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])


class HybridOnlineReliabilityTests(unittest.TestCase):
    @staticmethod
    def _runner(**decision_overrides) -> SimulationRunner:
        return SimulationRunner(
            SimulationConfig(
                decision=DecisionConfig(
                    hybrid_fusion_mode="adaptive_confidence",
                    **decision_overrides,
                ),
            )
        )

    @staticmethod
    def _override_decision(success_estimate_delay_s: float) -> dict:
        task = Task(
            task_id="t",
            vehicle_id="v",
            compute_cycles=1e9,
            data_size_mb=10.0,
            deadline_s=1.0,
            urgency=1.0,
            created_step=0,
        )
        baseline = OffloadEstimate(
            action=OffloadAction.V2I,
            delay_s=success_estimate_delay_s,
            energy_j=1.0,
            payment=0.1,
        )
        return {
            "task": task,
            "used_dqn": True,
            "action": OffloadAction.LOCAL,
            "baseline_action": OffloadAction.V2I,
            "candidate_map": {OffloadAction.V2I: baseline},
        }

    def test_reliability_counts_only_success_flipping_overrides(self):
        runner = self._runner(
            hybrid_online_reliability="always",
            hybrid_reliability_decay=0.5,
        )
        try:
            self.assertEqual(runner._hybrid_online_reliability(), 1.0)
            # Neutral: task succeeded and the game action was also estimated
            # on time.  Learned congestion avoidance must not be penalized.
            runner._record_override_outcome(self._override_decision(0.5), True)
            self.assertEqual(runner._hybrid_online_reliability(), 1.0)
            # Harmful: the override failed while the game action was
            # estimated on time.
            runner._record_override_outcome(self._override_decision(0.5), False)
            self.assertAlmostEqual(runner._hybrid_online_reliability(), 0.5)
            # Beneficial: the override succeeded while the game action was
            # estimated late.
            runner._record_override_outcome(self._override_decision(2.0), True)
            self.assertAlmostEqual(runner._hybrid_online_reliability(), 2.0 / 3.0)
            runner._decay_override_outcomes()
            self.assertAlmostEqual(
                runner._hybrid_online_reliability(), 1.5 / 2.0
            )
        finally:
            runner.mobility.close()

    def test_reliability_floor_and_disabled_flag(self):
        floored = self._runner(
            hybrid_online_reliability="always",
            hybrid_reliability_floor=0.25,
        )
        try:
            for _ in range(50):
                floored._record_override_outcome(
                    self._override_decision(0.5), False
                )
            self.assertEqual(floored._hybrid_online_reliability(), 0.25)
        finally:
            floored.mobility.close()
        disabled = self._runner()
        try:
            disabled._record_override_outcome(
                self._override_decision(0.5), False
            )
            self.assertEqual(disabled._hybrid_online_reliability(), 1.0)
            self.assertEqual(disabled._override_harmful_weight, 0.0)
        finally:
            disabled.mobility.close()

    @staticmethod
    def _follow_decision() -> dict:
        decision = HybridOnlineReliabilityTests._override_decision(0.5)
        decision["action"] = decision["baseline_action"]
        decision["used_dqn"] = False
        return decision

    def test_adequacy_tracks_game_follow_outcomes_and_blends_defense(self):
        runner = self._runner(
            hybrid_online_reliability="always",
            hybrid_game_adequacy_arbitration="always",
            hybrid_adequacy_defense_floor=0.5,
            hybrid_adequacy_defense_full=1.0,
        )
        try:
            self.assertEqual(runner._game_adequacy(), 1.0)
            self.assertEqual(runner._arbitration_game_adequacy(), 1.0)
            # Build refutation evidence: raw reliability drops to 0.5.
            runner._record_override_outcome(self._override_decision(0.5), False)
            self.assertAlmostEqual(runner._hybrid_online_reliability(), 0.5)
            # Game-follow failures reduce adequacy; with A = 0.5 the defense
            # weight reaches zero and the raw reliability is blended out.
            runner._record_override_outcome(self._follow_decision(), False)
            self.assertAlmostEqual(runner._game_adequacy(), 0.5)
            self.assertAlmostEqual(runner._hybrid_online_reliability(), 1.0)
            # Restoring adequacy restores a partial defense: with one failure
            # and three successes A = 0.8 and the weight is 0.6.
            for _ in range(3):
                runner._record_override_outcome(self._follow_decision(), True)
            self.assertAlmostEqual(runner._game_adequacy(), 0.8)
            self.assertAlmostEqual(
                runner._hybrid_online_reliability(), 1.0 - 0.6 * 0.5
            )
        finally:
            runner.mobility.close()

    def test_adequacy_disabled_keeps_full_defense_and_unit_damping(self):
        runner = self._runner(hybrid_online_reliability="always")
        try:
            runner._record_override_outcome(self._follow_decision(), False)
            self.assertEqual(runner._game_follow_failure_weight, 1.0)
            self.assertEqual(runner._arbitration_game_adequacy(), 1.0)
            runner._record_override_outcome(self._override_decision(0.5), False)
            self.assertAlmostEqual(runner._hybrid_online_reliability(), 0.5)
        finally:
            runner.mobility.close()


if __name__ == "__main__":
    unittest.main()
