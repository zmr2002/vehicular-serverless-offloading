from __future__ import annotations

from pathlib import Path
import unittest

from vehicular_offloading.config import DQNConfig, NetworkConfig, SimulationConfig


class ConfigTests(unittest.TestCase):
    def test_paper_profile_separates_steps_and_vehicles(self):
        config = SimulationConfig.from_toml(Path("configs/paper.toml"))
        self.assertEqual(config.steps, 2_000)
        self.assertEqual(config.vehicle_count, 2_000)
        self.assertEqual(config.dqn.hidden_sizes, (256, 128))
        self.assertEqual(config.dqn.state_size, 20)
        self.assertEqual(
            config.service_positions,
            (
                (1083.33, 1525.0),
                (1650.0, 1525.0),
                (2216.67, 1525.0),
                (1083.33, 2175.0),
                (1650.0, 2175.0),
                (2216.67, 2175.0),
            ),
        )
        self.assertEqual(config.serverless.concurrency_target, 10)
        self.assertEqual(config.serverless.max_instances, 10)
        self.assertEqual(config.vehicle_compute_hz, 2.0e9)
        self.assertEqual(config.service_compute_hz, config.vehicle_compute_hz)
        self.assertEqual(config.task_deadline_distribution, "uniform")
        self.assertEqual((config.task_deadline_min_s, config.task_deadline_max_s), (2.0, 3.0))
        self.assertEqual(config.network.v2v_channel_bandwidth_mhz, 50.0)
        self.assertFalse(config.decision.deadline_action_masking)
        self.assertFalse(config.decision.hybrid_objective_guidance)

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            SimulationConfig(strategy="unknown")

    def test_rejects_sequential_decision_timing(self):
        with self.assertRaises(ValueError):
            SimulationConfig(decision_timing="sequential")

    def test_improved_profile_is_separate_and_reviewable(self):
        baseline = SimulationConfig.from_toml(Path("configs/paper.toml"))
        improved = SimulationConfig.from_toml(Path("configs/paper-improved.toml"))
        self.assertEqual(len(improved.service_positions), 6)
        self.assertNotEqual(improved.service_positions, baseline.service_positions)
        self.assertEqual(improved.dqn.training_interval, 32)
        self.assertEqual(improved.dqn.intraop_threads, 1)
        self.assertEqual(improved.dqn.hidden_sizes, (256, 128))
        self.assertTrue(improved.decision.deadline_action_masking)
        self.assertTrue(improved.decision.hybrid_objective_guidance)
        self.assertFalse(improved.decision.hybrid_cloud_capacity_guard)
        self.assertFalse(baseline.decision.hybrid_cloud_capacity_guard)
        self.assertEqual(improved.cloud_pricing_mode, "leader_best_response")
        self.assertEqual(baseline.cloud_pricing_mode, "queue")
        self.assertEqual(improved.task_compute_distribution, "uniform")
        self.assertEqual(improved.task_probability, 0.6)
        self.assertEqual(baseline.task_compute_distribution, "discrete")
        self.assertEqual(improved.decision.hybrid_residual_weight, 0.75)
        self.assertEqual(improved.decision.hybrid_fusion_mode, "residual")
        self.assertTrue(improved.decision.stackelberg_deadline_action_masking)
        self.assertEqual(improved.decision.stackelberg_on_time_bonus, 0.5)
        self.assertTrue(improved.decision.hybrid_residual_congestion_adaptation)
        self.assertEqual(improved.decision.hybrid_cloud_guard_ratio, 1.0)
        self.assertEqual(improved.service_role_mode, "dynamic_idle")
        self.assertEqual(baseline.service_role_mode, "dynamic_idle")

    def test_follower_game_profile_keeps_thesis_dqn_and_autonomous_fusion(self):
        config = SimulationConfig.from_toml(
            Path("configs/paper-follower-game.toml")
        )
        self.assertEqual(config.cloud_pricing_mode, "follower_best_response")
        self.assertEqual(config.dqn.hidden_sizes, (256, 128))
        self.assertGreater(config.dqn.game_guidance_weight, 0.0)
        self.assertEqual(
            config.decision.hybrid_fusion_mode,
            "adaptive_confidence",
        )
        self.assertGreater(
            config.decision.hybrid_game_confidence_threshold,
            0.0,
        )
        self.assertGreater(
            config.decision.hybrid_dqn_opposition_threshold,
            0.0,
        )
        self.assertEqual(config.cloud_price_response_iterations, 3)
        self.assertEqual(config.cloud_price_response_min_iterations, 1)
        self.assertEqual(config.cloud_price_response_relaxation, 0.5)
        self.assertEqual(config.cloud_price_response_policy, "softmax")
        self.assertEqual(config.cloud_leader_late_tolerance, 0.02)
        self.assertFalse(config.cloud_price_state_consistency)
        self.assertTrue(config.cloud_price_batch_candidates)
        self.assertEqual(config.cloud_price_outer_iterations, 2)
        self.assertEqual(config.cloud_price_outer_min_iterations, 2)
        self.assertEqual(config.cloud_price_outer_tolerance, 0.0)
        self.assertTrue(config.decision.synchronous_v2v_queue_forecast)

    def test_pre_serverless_profile_freezes_adaptive_gate_semantics(self):
        config = SimulationConfig.from_toml(
            Path("configs/pre-serverless-adaptive-gate.toml")
        )
        self.assertEqual(config.decision.hybrid_fusion_mode, "adaptive_confidence")
        self.assertFalse(config.decision.synchronous_v2v_queue_forecast)
        self.assertFalse(config.cloud_price_state_consistency)
        self.assertEqual(config.cloud_leader_late_tolerance, 0.0)

    def test_state_consistent_and_fixed_point_profiles_are_separate(self):
        baseline = SimulationConfig.from_toml(
            Path("configs/paper-thesis-hybrid.toml")
        )
        consistent = SimulationConfig.from_toml(
            Path("configs/paper-thesis-hybrid-state-consistent.toml")
        )
        fixed_point = SimulationConfig.from_toml(
            Path("configs/paper-thesis-hybrid-fixed-point.toml")
        )
        argmax = SimulationConfig.from_toml(
            Path("configs/paper-thesis-hybrid-argmax.toml")
        )
        self.assertFalse(baseline.cloud_price_state_consistency)
        self.assertTrue(consistent.cloud_price_state_consistency)
        self.assertEqual(fixed_point.cloud_price_response_iterations, 5)
        self.assertEqual(fixed_point.cloud_price_response_tolerance, 0.01)
        self.assertEqual(fixed_point.cloud_price_outer_iterations, 5)
        self.assertEqual(fixed_point.cloud_price_outer_tolerance, 0.01)
        self.assertEqual(argmax.cloud_price_response_policy, "argmax")

    def test_thesis_hybrid_profile_inherits_physics_but_delegates_to_dqn(self):
        enhanced = SimulationConfig.from_toml(
            Path("configs/paper-follower-game.toml")
        )
        thesis = SimulationConfig.from_toml(
            Path("configs/paper-thesis-hybrid.toml")
        )
        self.assertEqual(thesis.task_probability, enhanced.task_probability)
        self.assertEqual(thesis.service_positions, enhanced.service_positions)
        self.assertEqual(
            thesis.network.v2v_channel_bandwidth_mhz,
            enhanced.network.v2v_channel_bandwidth_mhz,
        )
        self.assertEqual(thesis.decision.hybrid_fusion_mode, "delegated")
        self.assertEqual(thesis.dqn.game_guidance_weight, 0.0)
        self.assertTrue(thesis.decision.synchronous_v2v_queue_forecast)

    def test_rejects_invalid_offload_compression(self):
        with self.assertRaisesRegex(ValueError, "compression"):
            SimulationConfig(offload_compression_ratio=0.0)

    def test_rejects_non_thesis_network_shape(self):
        with self.assertRaisesRegex(ValueError, "20->256->128->3"):
            SimulationConfig(dqn=DQNConfig(hidden_sizes=(128, 64)))

    def test_rejects_invalid_numeric_configuration(self):
        with self.assertRaisesRegex(ValueError, "bandwidths"):
            SimulationConfig(network=NetworkConfig(v2v_channel_bandwidth_mhz=0))
        with self.assertRaisesRegex(ValueError, "minimum_free_disk_gb"):
            SimulationConfig(minimum_free_disk_gb=-1.0)
        with self.assertRaisesRegex(ValueError, "task_record_sample_rate"):
            SimulationConfig(task_record_sample_rate=1.01)
        with self.assertRaisesRegex(ValueError, "decision_trace_path"):
            SimulationConfig(decision_trace_mode="record")

    def test_rejects_unknown_cloud_pricing_mode(self):
        with self.assertRaisesRegex(ValueError, "cloud_pricing_mode"):
            SimulationConfig(cloud_pricing_mode="unknown")

    def test_rejects_invalid_follower_response_controls(self):
        with self.assertRaisesRegex(ValueError, "min_iterations"):
            SimulationConfig(
                cloud_price_response_iterations=2,
                cloud_price_response_min_iterations=3,
            )
        with self.assertRaisesRegex(ValueError, "response_policy"):
            SimulationConfig(cloud_price_response_policy="unknown")
        with self.assertRaisesRegex(ValueError, "outer_min_iterations"):
            SimulationConfig(
                cloud_price_outer_iterations=2,
                cloud_price_outer_min_iterations=3,
            )
        with self.assertRaisesRegex(ValueError, "outer_tolerance"):
            SimulationConfig(cloud_price_outer_tolerance=-0.1)

    def test_rejects_invalid_task_compute_distribution(self):
        with self.assertRaisesRegex(ValueError, "task_compute_distribution"):
            SimulationConfig(task_compute_distribution="bimodal")

    def test_rejects_invalid_task_deadline_distribution(self):
        with self.assertRaisesRegex(ValueError, "task_deadline_distribution"):
            SimulationConfig(task_deadline_distribution="bimodal")
        with self.assertRaisesRegex(ValueError, "deadline bounds"):
            SimulationConfig(task_deadline_min_s=3.0, task_deadline_max_s=2.0)

    def test_dynamic_idle_rejects_role_dependent_vehicle_cpu(self):
        with self.assertRaisesRegex(ValueError, "same physical vehicles"):
            SimulationConfig(
                service_role_mode="dynamic_idle",
                vehicle_compute_hz=2.0e9,
                service_compute_hz=10.0e9,
            )

        dedicated = SimulationConfig(
            service_role_mode="fixed_ratio",
            vehicle_compute_hz=2.0e9,
            service_compute_hz=10.0e9,
        )
        self.assertEqual(dedicated.service_compute_hz, 10.0e9)

    def test_evaluation_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            SimulationConfig(dqn=DQNConfig(mode="evaluate"))


if __name__ == "__main__":
    unittest.main()
