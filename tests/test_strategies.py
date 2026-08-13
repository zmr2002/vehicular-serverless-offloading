from __future__ import annotations

import unittest
import random

import torch

from vehicular_offloading.config import DecisionConfig, DQNConfig, NetworkConfig, SimulationConfig
from vehicular_offloading.domain import OffloadAction, OffloadEstimate, Task, VehicleState
from vehicular_offloading.dqn import DQNAgent
from vehicular_offloading.pricing import (
    cloud_leader_price,
    cloud_price,
    cloud_price_candidates,
    evaluate_cloud_leader_response,
    service_quote,
)
from vehicular_offloading.strategies import (
    DecisionContext,
    choose_action,
    estimates,
    game_guidance,
    policy_action_ids,
    reward_for,
    v2i_estimate,
)


class StrategyTests(unittest.TestCase):
    def test_cloud_price_increases_with_queue(self):
        self.assertGreater(cloud_price(0.1, 10, 10), cloud_price(0.1, 0, 10))

    def test_cloud_leader_response_is_bounded_and_capacity_aware(self):
        low = cloud_leader_price(
            0.1, 0.3, 0.1, 0.3, 3.0, 0.1, 0, 10, 0.1, 0.85, 1.0, 0.05, 1.0
        )
        high = cloud_leader_price(
            0.1, 0.3, 0.1, 0.3, 3.0, 0.1, 20, 10, 1.0, 0.85, 1.0, 0.05, 1.0
        )
        bounded = cloud_leader_price(
            1.0, 1.0, 0.1, 0.3, 3.0, 1.0, 100, 10, 2.0, 0.85, 10.0, 0.05, 1.0
        )
        capacity_only = cloud_leader_price(
            0.1, 0.3, 0.1, 0.3, 3.0, 0.1, 0, 10, 1.0, 0.85, 1.0, 0.05, 1.0
        )
        self.assertGreater(high, low)
        self.assertGreater(capacity_only, low)
        self.assertEqual(bounded, 1.0)

    def test_follower_price_candidates_are_bounded_and_keep_anchors(self):
        candidates = cloud_price_candidates(0.05, 1.0, 5, (0.1, 2.0))
        self.assertEqual(candidates[0], 0.05)
        self.assertEqual(candidates[-1], 1.0)
        self.assertIn(0.1, candidates)

    def test_follower_leader_utility_penalizes_overload_and_late_admission(self):
        balanced = evaluate_cloud_leader_response(
            0.5, 20.0, 100.0, 0.0, 10, 0.3, 0.1, 1.0, 1.0, 1.0
        )
        overloaded = evaluate_cloud_leader_response(
            0.5, 80.0, 100.0, 5.0, 10, 0.3, 0.1, 1.0, 1.0, 1.0
        )
        self.assertGreater(balanced.leader_score, overloaded.leader_score)
        self.assertGreater(overloaded.capacity_penalty, 0.0)
        self.assertGreater(overloaded.timeout_penalty, 0.0)

    def test_follower_leader_utility_preserves_revenue_objective(self):
        lower = evaluate_cloud_leader_response(
            0.4,
            20.0,
            100.0,
            0.1,
            10,
            0.3,
            0.1,
            1.0,
            1.0,
            1.0,
            predicted_cloud_requests=2.0,
            late_tolerance=0.02,
        )
        higher = evaluate_cloud_leader_response(
            0.8,
            20.0,
            100.0,
            0.1,
            10,
            0.3,
            0.1,
            1.0,
            1.0,
            1.0,
            predicted_cloud_requests=2.0,
            late_tolerance=0.02,
        )
        self.assertGreater(higher.revenue_score, lower.revenue_score)
        self.assertGreater(higher.leader_score, lower.leader_score)
        self.assertEqual(lower.capacity_violation, 0.0)
        self.assertEqual(lower.timeout_violation, 0.0)

    def test_service_vehicle_quote_responds_to_anticipated_demand(self):
        vehicle = VehicleState("service", (0.0, 0.0), 0.0, 2e9)
        low = service_quote(vehicle, 0.5, 0.5, 0.05, 5)
        high = service_quote(
            vehicle,
            0.5,
            0.5,
            0.05,
            5,
            anticipated_demand_ratio=1.5,
            demand_price_weight=0.2,
        )
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreater(high.price, low.price)
        self.assertLessEqual(high.price, 0.5)

    def test_service_vehicle_rejects_quote_below_reservation_utility(self):
        vehicle = VehicleState(
            "service",
            (0.0, 0.0),
            0.0,
            2e9,
            energy_level=0.1,
        )
        quote = service_quote(vehicle, 0.01, 0.5, 0.05, 5)
        self.assertIsNone(quote)

    def test_success_reward_and_timeout_penalty(self):
        config = SimulationConfig()
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        success_reward, success = reward_for(OffloadEstimate(OffloadAction.LOCAL, 0.5, 1.0, 0.0), task, config)
        timeout_reward, timeout_success = reward_for(
            OffloadEstimate(OffloadAction.LOCAL, 2.0, 1.0, 0.0), task, config
        )
        self.assertTrue(success)
        self.assertFalse(timeout_success)
        self.assertGreater(success_reward, 0.0)
        self.assertLess(timeout_reward, 0.0)

    def test_game_guidance_reports_a_normalized_best_response_margin(self):
        config = SimulationConfig()
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(
                OffloadAction.LOCAL, 0.5, 100.0, 0.0
            ),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), False
            ),
            OffloadAction.V2I: OffloadEstimate(
                OffloadAction.V2I, 0.8, 1.0, 0.1
            ),
        }
        guidance = game_guidance(context, candidates, config, [0, 2])
        self.assertEqual(guidance.action, OffloadAction.LOCAL)
        self.assertGreater(guidance.confidence, 0.0)

    def test_dqn_reward_uses_the_same_normalized_cost_as_game_objective(self):
        config = SimulationConfig()
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        estimate = OffloadEstimate(OffloadAction.V2I, 1.0, 100.0, 0.5)
        expected_cost = 1.0 / 2.0 + 0.2 * 100.0 / 250.0 + 0.075 * 0.5 / 0.5
        reward, success = reward_for(estimate, task, config)
        self.assertTrue(success)
        self.assertAlmostEqual(
            reward,
            config.reward.on_time_bonus - config.reward.cost_scale * expected_cost,
        )

    def test_dqn_cannot_choose_an_avoidable_timeout(self):
        config = SimulationConfig(dqn=DQNConfig(epsilon_start=0.0, epsilon_end=0.0))
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        vehicle = VehicleState("v", (1083.33, 1525.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(1083.33, 1525.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 50.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 2.0, 1.0, 0.1),
        }
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        action, _, _ = choose_action(
            "dqn", context, candidates, config, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.LOCAL)

        all_late = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 2.5, 50.0, 0.0),
            OffloadAction.V2V: candidates[OffloadAction.V2V],
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.2, 1.0, 0.1),
        }
        self.assertEqual(
            policy_action_ids("dqn", context, all_late, config),
            [int(OffloadAction.LOCAL), int(OffloadAction.V2I)],
        )
        action, _, _ = choose_action(
            "dqn", context, all_late, config, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.V2I)

    def test_thesis_dqn_action_space_does_not_apply_deadline_mask(self):
        config = SimulationConfig(
            dqn=DQNConfig(epsilon_start=0.0, epsilon_end=0.0),
            decision=DecisionConfig(deadline_action_masking=False),
        )
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        vehicle = VehicleState("v", (1083.33, 1525.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(1083.33, 1525.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 50.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 2.0, 1.0, 0.1),
        }
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        action, _, _ = choose_action(
            "dqn", context, candidates, config, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.V2I)

    def test_hybrid_guidance_rejects_unsafe_dqn_deviation(self):
        config = SimulationConfig(dqn=DQNConfig(epsilon_start=0.0, epsilon_end=0.0))
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (1083.33, 1525.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(1083.33, 1525.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 50.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.8, 1.0, 1.0),
        }
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg", context, candidates, config, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.LOCAL)
        self.assertTrue(used_dqn)

    def test_adaptive_hybrid_keeps_confident_game_action_below_capacity(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
                hybrid_game_confidence_threshold=0.15,
                hybrid_dqn_opposition_threshold=1.0,
                hybrid_congestion_sensitivity=1.0,
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=0.5,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.V2I)
        self.assertFalse(used_dqn)

    def test_adaptive_hybrid_defers_to_dqn_for_cloud_queue_externality(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
                hybrid_game_confidence_threshold=0.15,
                hybrid_dqn_opposition_threshold=0.15,
                hybrid_congestion_sensitivity=1.0,
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=1.7,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.LOCAL)
        self.assertTrue(used_dqn)

    def test_adaptive_hybrid_keeps_thesis_strict_dominance_rule(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=1.7,
        )
        candidates[OffloadAction.V2I] = OffloadEstimate(
            OffloadAction.V2I,
            0.5,
            1.0,
            0.0,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.V2I)
        self.assertFalse(used_dqn)

    def test_thesis_hybrid_delegates_non_dominant_choices_to_dqn(self):
        config = SimulationConfig(
            dqn=DQNConfig(epsilon_start=0.0, epsilon_end=0.0),
            decision=DecisionConfig(
                hybrid_fusion_mode="delegated",
                hybrid_objective_guidance=False,
                hybrid_cloud_capacity_guard=False,
            ),
        )
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (1083.33, 1525.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(1083.33, 1525.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 50.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.8, 1.0, 1.0),
        }
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.V2I)
        self.assertTrue(used_dqn)

    def test_stackelberg_uses_soft_rather_than_hard_deadline_utility(self):
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.8, 500.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.01, 0.0, 0.0),
        }
        agent = DQNAgent(DQNConfig(epsilon_start=0.0, epsilon_end=0.0), seed=5)
        soft = SimulationConfig(
            decision=DecisionConfig(
                stackelberg_deadline_action_masking=False,
                stackelberg_on_time_bonus=0.0,
            )
        )
        action, _, _ = choose_action(
            "stackelberg", context, candidates, soft, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.V2I)

        rewarded = SimulationConfig(
            decision=DecisionConfig(
                stackelberg_deadline_action_masking=False,
                stackelberg_on_time_bonus=0.5,
            )
        )
        action, _, _ = choose_action(
            "stackelberg", context, candidates, rewarded, random.Random(1), agent, explore=False
        )
        self.assertEqual(action, OffloadAction.LOCAL)

    def test_delegated_hybrid_sends_non_dominated_choice_to_dqn(self):
        config = SimulationConfig(
            dqn=DQNConfig(epsilon_start=0.0, epsilon_end=0.0),
            decision=DecisionConfig(
                hybrid_fusion_mode="delegated",
            ),
        )
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 10.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.8, 1.0, 1.0),
        }
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.V2I)
        self.assertTrue(used_dqn)

    def test_hybrid_rule_decision_advances_exploration_once(self):
        config = SimulationConfig(
            dqn=DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5)
        )
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.5, 10.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 1.8, 100.0, 1.0),
        }
        agent = DQNAgent(config.dqn, seed=5)
        _, _, used_dqn = choose_action(
            "hybrid_stackelberg", context, candidates, config, random.Random(1), agent
        )
        self.assertFalse(used_dqn)
        self.assertEqual(agent.epsilon, 0.5)

    def test_hybrid_reserves_saturated_cloud_for_urgent_tasks(self):
        config = SimulationConfig(
            decision=DecisionConfig(
                hybrid_cloud_capacity_guard=True,
                hybrid_cloud_guard_ratio=1.0,
            )
        )
        task = Task("t", "v", 1e9, 1.0, 1.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=10,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(OffloadAction.LOCAL, 0.8, 50.0, 0.0),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V, float("inf"), float("inf"), float("inf"), feasible=False
            ),
            OffloadAction.V2I: OffloadEstimate(OffloadAction.V2I, 0.4, 1.0, 0.1),
        }
        self.assertEqual(
            policy_action_ids("hybrid_stackelberg", context, candidates, config),
            [int(OffloadAction.LOCAL)],
        )

        candidates[OffloadAction.LOCAL] = OffloadEstimate(
            OffloadAction.LOCAL, 1.2, 50.0, 0.0
        )
        self.assertEqual(
            policy_action_ids("hybrid_stackelberg", context, candidates, config),
            [int(OffloadAction.V2I)],
        )

    def test_v2i_energy_includes_transmission_and_cloud_computation(self):
        config = SimulationConfig(network=NetworkConfig(channel_capacity_model="distance_only"))
        task = Task("t", "v", 1e9, 0.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        v2i = estimates(context, config)[OffloadAction.V2I]
        self.assertAlmostEqual(v2i.energy_j, 1.6)

    def test_standalone_v2i_estimate_matches_full_candidate_estimation(self):
        config = SimulationConfig(
            offload_compression_ratio=0.5,
            compression_cycles_per_mb=2.0e6,
            network=NetworkConfig(channel_capacity_model="distance_only"),
        )
        task = Task("t", "v", 3e9, 80.0, 4.0, 0.5, 0)
        vehicle = VehicleState("v", (20.0, 30.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.4,
            cloud_queue_length=7,
            cloud_platform_overhead_s=0.03,
            server_position=(120.0, 230.0),
            price_history=(),
            adjacency={"v": ()},
        )
        self.assertEqual(
            v2i_estimate(context, config),
            estimates(context, config)[OffloadAction.V2I],
        )

    def test_local_backlog_adds_delay_without_charging_waiting_energy(self):
        config = SimulationConfig()
        task = Task("t", "v", 1e9, 1.0, 3.0, 0.5, 0)
        vehicle = VehicleState(
            "v", (0.0, 0.0), 0.0, 2e9, workload_cycles=1e9
        )
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        local = estimates(context, config)[OffloadAction.LOCAL]
        self.assertAlmostEqual(local.delay_s, 1.0)
        self.assertAlmostEqual(local.energy_j, 50.0)

    @staticmethod
    def _adaptive_hybrid_cloud_case(
        config: SimulationConfig,
        *,
        cloud_capacity_ratio: float,
    ) -> tuple[DecisionContext, dict[OffloadAction, OffloadEstimate]]:
        task = Task("t", "v", 1e9, 1.0, 2.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=10,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
            cloud_capacity_ratio=cloud_capacity_ratio,
        )
        candidates = {
            OffloadAction.LOCAL: OffloadEstimate(
                OffloadAction.LOCAL,
                1.0,
                50.0,
                0.0,
            ),
            OffloadAction.V2V: OffloadEstimate(
                OffloadAction.V2V,
                float("inf"),
                float("inf"),
                float("inf"),
                feasible=False,
            ),
            OffloadAction.V2I: OffloadEstimate(
                OffloadAction.V2I,
                0.5,
                80.0,
                0.0,
            ),
        }
        return context, candidates

    def test_compression_reduces_offload_delay_and_accounts_for_preprocessing(self):
        network = NetworkConfig(channel_capacity_model="distance_only")
        base = SimulationConfig(network=network)
        compressed = SimulationConfig(
            offload_compression_ratio=0.5,
            compression_cycles_per_mb=2.0e6,
            network=NetworkConfig(channel_capacity_model="distance_only"),
        )
        task = Task("t", "v", 1e9, 100.0, 5.0, 0.5, 0)
        vehicle = VehicleState("v", (0.0, 0.0), 10.0, 2e9)
        context = DecisionContext(
            task=task,
            vehicle=vehicle,
            vehicles={"v": vehicle},
            service_quotes={},
            cloud_price=0.1,
            cloud_queue_length=0,
            cloud_platform_overhead_s=0.0,
            server_position=(0.0, 0.0),
            price_history=(),
            adjacency={"v": ()},
        )
        original = estimates(context, base)[OffloadAction.V2I]
        reduced = estimates(context, compressed)[OffloadAction.V2I]
        self.assertAlmostEqual(original.delay_s, 100.0 * 8.0 / 170.0 + 0.02)
        self.assertAlmostEqual(reduced.delay_s, 0.1 + 50.0 * 8.0 / 170.0 + 0.02)
        self.assertGreater(reduced.energy_j, 40.0)
        self.assertLess(reduced.delay_s, original.delay_s)

    def test_adaptive_hybrid_online_reliability_suppresses_dqn_override(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
                hybrid_game_confidence_threshold=0.15,
                hybrid_dqn_opposition_threshold=0.15,
                hybrid_congestion_sensitivity=1.0,
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=1.7,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
            online_reliability=0.0,
        )
        self.assertEqual(action, OffloadAction.V2I)
        self.assertFalse(used_dqn)

    def test_low_game_adequacy_hands_authority_to_the_dqn(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
                hybrid_game_confidence_threshold=0.15,
                hybrid_dqn_opposition_threshold=1.0,
                hybrid_congestion_sensitivity=1.0,
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=0.5,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        confident_game, _, game_used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
            game_adequacy=1.0,
        )
        self.assertEqual(confident_game, OffloadAction.V2I)
        self.assertFalse(game_used_dqn)
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
            game_adequacy=0.5,
        )
        self.assertEqual(action, OffloadAction.LOCAL)
        self.assertTrue(used_dqn)

    def test_game_evidence_cap_bounds_congestion_inflated_margins(self):
        config = SimulationConfig(
            dqn=DQNConfig(
                mode="evaluate",
                checkpoint_path="unused.pt",
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
            decision=DecisionConfig(
                hybrid_fusion_mode="adaptive_confidence",
                hybrid_game_confidence_threshold=0.15,
                hybrid_dqn_opposition_threshold=1.0,
                hybrid_congestion_sensitivity=1.0,
                hybrid_game_evidence_cap=0.5,
            ),
        )
        context, candidates = self._adaptive_hybrid_cloud_case(
            config,
            cloud_capacity_ratio=0.5,
        )
        agent = DQNAgent(config.dqn, seed=5)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(
                torch.tensor([100.0, 0.0, 0.0])
            )
        action, _, used_dqn = choose_action(
            "hybrid_stackelberg",
            context,
            candidates,
            config,
            random.Random(1),
            agent,
            explore=False,
        )
        self.assertEqual(action, OffloadAction.LOCAL)
        self.assertTrue(used_dqn)


if __name__ == "__main__":
    unittest.main()
