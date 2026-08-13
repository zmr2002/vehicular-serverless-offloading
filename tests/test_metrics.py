from __future__ import annotations

import unittest

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.domain import OffloadAction, OffloadResult
from vehicular_offloading.metrics import MetricsAccumulator


class MetricsTests(unittest.TestCase):
    def test_exact_diagnostics_preserve_local_action_and_window_counts(self):
        metrics = MetricsAccumulator()
        metrics.observe_step(10, 4, 6, 8e9, step=250)
        metrics.add(
            OffloadResult(
                task_id="diagnostic",
                vehicle_id="source",
                action=OffloadAction.V2I,
                delay_s=1.0,
                energy_j=2.0,
                payment=0.5,
                reward=3.0,
                success=True,
                step=250,
                oracle_success=True,
                used_dqn=True,
                game_action=OffloadAction.LOCAL,
                stackelberg_action=OffloadAction.V2I,
                dqn_action=OffloadAction.V2I,
                dqn_q_margin=0.75,
                cloud_queue_length=2,
                predicted_cloud_capacity_ratio=0.4,
                cloud_price=0.2,
                hybrid_deviation=True,
                hybrid_deviation_beneficial=True,
                hybrid_decision_source="game_gate",
            )
        )
        diagnostics = metrics.diagnostics(completed_steps=251)
        window = diagnostics["windows"][0]
        self.assertEqual(window["window_index"], 1)
        self.assertEqual(window["task_count"], 1)
        self.assertEqual(window["avg_dqn_q_margin"], 0.75)
        self.assertEqual(window["game_to_dqn"]["local"]["v2i"], 1)
        self.assertEqual(window["game_to_final"]["local"]["v2i"], 1)
        self.assertEqual(window["dqn_to_final"]["v2i"]["v2i"], 1)

    def test_all_late_cloud_admission_reports_wasted_capacity(self):
        config = SimulationConfig(steps=2, cloud_compute_hz=50e9)
        metrics = MetricsAccumulator(config.vehicle_compute_hz)
        metrics.add(
            OffloadResult(
                task_id="late-cloud",
                vehicle_id="source",
                action=OffloadAction.V2I,
                delay_s=4.0,
                energy_j=1.0,
                payment=0.1,
                reward=-1.0,
                success=False,
                step=0,
                task_compute_cycles=5e9,
                task_deadline_s=2.0,
                all_actions_late=True,
            )
        )
        metrics.add(
            OffloadResult(
                task_id="late-local",
                vehicle_id="source",
                action=OffloadAction.LOCAL,
                delay_s=3.0,
                energy_j=1.0,
                payment=0.0,
                reward=-1.0,
                success=False,
                step=1,
                task_compute_cycles=3e9,
                task_deadline_s=2.0,
                all_actions_late=True,
            )
        )
        summary = metrics.summary(
            config,
            completed_steps=2,
            realized_vehicle_count=1,
            peak_active_vehicles=1,
            replay_size=0,
            dqn_transitions=0,
            dqn_updates=0,
            final_epsilon=0.0,
        )
        self.assertEqual(summary.all_actions_late_rate, 1.0)
        self.assertEqual(summary.all_late_cloud_admission_rate, 0.5)
        self.assertEqual(summary.avg_all_late_cloud_cycles_per_step, 2.5e9)
        self.assertAlmostEqual(summary.all_late_cloud_to_capacity_ratio, 0.005)

    def test_queue_load_and_v2v_rescue_are_measured_separately(self):
        config = SimulationConfig(
            steps=1,
            vehicle_count=10,
            service_role_mode="dynamic_idle",
            vehicle_compute_hz=2e9,
            service_compute_hz=2e9,
        )
        metrics = MetricsAccumulator(config.vehicle_compute_hz)
        metrics.observe_step(
            active_vehicle_count=10,
            task_vehicle_count=6,
            service_vehicle_count=4,
            arrived_cycles=18e9,
        )
        metrics.add(
            OffloadResult(
                task_id="task",
                vehicle_id="source",
                action=OffloadAction.V2V,
                delay_s=1.4,
                energy_j=1.0,
                payment=0.1,
                reward=1.0,
                success=True,
                step=0,
                task_compute_cycles=1e9,
                task_deadline_s=1.5,
                local_estimate_s=2.0,
                v2v_estimate_s=1.4,
                v2i_estimate_s=1.0,
                source_workload_s=1.5,
                v2v_target_workload_s=0.5,
            )
        )
        summary = metrics.summary(
            config,
            completed_steps=1,
            realized_vehicle_count=10,
            peak_active_vehicles=10,
            replay_size=0,
            dqn_transitions=0,
            dqn_updates=0,
            final_epsilon=0.0,
        )
        self.assertAlmostEqual(summary.offered_vehicle_compute_load_ratio, 0.9)
        self.assertAlmostEqual(summary.task_vehicle_step_ratio, 0.6)
        self.assertAlmostEqual(summary.service_vehicle_step_ratio, 0.4)
        self.assertEqual(summary.queue_induced_local_timeout_ratio, 1.0)
        self.assertEqual(summary.v2v_latency_advantage_ratio, 1.0)
        self.assertEqual(summary.v2v_rescuable_task_ratio, 1.0)
        self.assertEqual(summary.intrinsic_local_infeasible_task_ratio, 0.0)
        self.assertEqual(summary.avg_source_workload_s, 1.5)
        self.assertEqual(summary.avg_v2v_target_workload_s, 0.5)

    def test_serverless_aggregates_do_not_depend_on_task_log_sampling(self):
        config = SimulationConfig(steps=1)
        metrics = MetricsAccumulator(config.vehicle_compute_hz)
        metrics.add(
            OffloadResult(
                task_id="http",
                vehicle_id="source",
                action=OffloadAction.V2I,
                delay_s=0.2,
                energy_j=0.1,
                payment=0.1,
                reward=1.0,
                success=False,
                step=0,
                preprocessing_delay_ms=1.0,
                radio_delay_ms=2.0,
                physical_compute_ms=50.0,
                physical_queue_ms=3.0,
                dispatch_queue_ms=4.0,
                platform_overhead_ms=10.0,
                total_delay_ms=70.0,
                client_latency_ms=15.0,
                http_latency_ms=11.0,
                scaled_processing_ms=1.0,
                http_attempts=2,
                http_retry_count=1,
                cold_start=True,
                instance_id="pod-one",
            )
        )
        summary = metrics.summary(
            config,
            completed_steps=1,
            realized_vehicle_count=1,
            peak_active_vehicles=1,
            replay_size=0,
            dqn_transitions=0,
            dqn_updates=0,
            final_epsilon=0.0,
        )
        self.assertEqual(summary.serverless_http_request_count, 1)
        self.assertEqual(summary.serverless_http_attempt_count, 2)
        self.assertEqual(summary.serverless_retried_request_count, 1)
        self.assertEqual(summary.serverless_http_retry_count, 1)
        self.assertEqual(summary.serverless_v2i_failure_count, 1)
        self.assertEqual(summary.serverless_cold_start_count, 1)
        self.assertEqual(summary.serverless_distinct_instance_count, 1)
        self.assertEqual(summary.p95_serverless_client_latency_ms, 15.0)
        self.assertEqual(summary.serverless_delay_decomposition_max_error_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
