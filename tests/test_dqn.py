from __future__ import annotations

import unittest
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from vehicular_offloading.config import DQNConfig
from vehicular_offloading.dqn import DQNAgent, QNetwork, ReplayBuffer


class DQNTests(unittest.TestCase):
    def test_ring_replay_preserves_chronological_sampling_order(self):
        replay = ReplayBuffer(capacity=3, seed=7)
        for index in range(5):
            state = np.full(20, index, dtype=np.float32)
            replay.add(state, 0, 0.0, state, False, np.ones(3, dtype=np.bool_))
        indices = random.Random(7).sample(range(3), 2)
        expected = [[2.0, 3.0, 4.0][index] for index in indices]
        actual = [float(item[0][0]) for item in replay.sample(2)]
        self.assertEqual(actual, expected)

    def test_load_stratified_replay_retains_each_congestion_regime(self):
        replay = ReplayBuffer(capacity=8, seed=7, sampling="load_stratified")
        for ratio in (0.1, 0.6, 1.0, 1.5):
            for index in range(20):
                state = np.full(20, index / 100.0, dtype=np.float32)
                state[10] = ratio
                replay.add(
                    state,
                    0,
                    0.0,
                    state,
                    False,
                    np.ones(3, dtype=np.bool_),
                )
        sampled_ratios = [float(item[0][10]) for item in replay.sample(8)]
        self.assertEqual(len(replay), 8)
        strata = [
            0 if value < 0.5 else 1 if value < 0.85 else 2 if value < 1.2 else 3
            for value in sampled_ratios
        ]
        self.assertEqual([strata.count(index) for index in range(4)], [2, 2, 2, 2])

    def test_load_stratified_replay_reuses_unclaimed_capacity(self):
        replay = ReplayBuffer(capacity=10, seed=8, sampling="load_stratified")
        for index in range(30):
            state = np.zeros(20, dtype=np.float32)
            state[10] = 0.2
            replay.add(
                state,
                0,
                float(index),
                state,
                False,
                np.ones(3, dtype=np.bool_),
            )
        rare = np.zeros(20, dtype=np.float32)
        rare[10] = 1.5
        replay.add(
            rare,
            2,
            99.0,
            rare,
            False,
            np.ones(3, dtype=np.bool_),
        )

        self.assertEqual(len(replay), 10)
        self.assertEqual(len(replay._strata[0]), 9)
        self.assertEqual(len(replay._strata[3]), 1)

    def test_architecture_matches_thesis(self):
        network = QNetwork()
        self.assertEqual(network.layers[0].in_features, 20)
        self.assertEqual(network.layers[0].out_features, 256)
        self.assertEqual(network.layers[2].out_features, 128)
        self.assertEqual(network.layers[4].out_features, 3)

    def test_replay_updates_weights_and_target(self):
        config = DQNConfig(
            replay_capacity=20,
            batch_size=4,
            warmup_transitions=4,
            target_update_interval=2,
            epsilon_start=0.2,
        )
        agent = DQNAgent(config, seed=7)
        initial = [parameter.detach().clone() for parameter in agent.online.parameters()]
        for index in range(8):
            state = np.full(20, index / 10.0, dtype=np.float32)
            next_state = state + 0.01
            agent.store(state, index % 3, float(index), next_state, index == 7)
        self.assertEqual(agent.transition_count, 8)
        self.assertIsNotNone(agent.update())
        self.assertIsNotNone(agent.update())
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(initial, agent.online.parameters())
            )
        )
        self.assertTrue(
            all(
                torch.equal(online, target)
                for online, target in zip(agent.online.parameters(), agent.target.parameters())
            )
        )

    def test_game_guidance_ranking_preserves_td_learning_and_orders_q_values(self):
        config = DQNConfig(
            replay_capacity=8,
            batch_size=4,
            warmup_transitions=4,
            gamma=0.0,
            learning_rate=0.01,
            game_guidance_weight=1.0,
            game_guidance_margin=1.0,
        )
        agent = DQNAgent(config, seed=11)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.target.load_state_dict(agent.online.state_dict())
        state = np.zeros(20, dtype=np.float32)
        for _ in range(4):
            agent.store(
                state,
                0,
                0.0,
                state,
                True,
                preferred_action=2,
                preference_weight=1.0,
            )
        self.assertIsNotNone(agent.update())
        q_values = agent.q_values(state)
        self.assertGreater(q_values[2], q_values[0])

    def test_action_mask_applies_to_exploration_and_greedy_selection(self):
        agent = DQNAgent(DQNConfig(epsilon_start=1.0), seed=17)
        state = np.zeros(20, dtype=np.float32)
        explored = {
            agent.choose_action(state, explore=True, allowed_actions=(0, 1))
            for _ in range(100)
        }
        self.assertEqual(explored, {0, 1})

        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([1.0, 2.0, 100.0]))
        self.assertEqual(agent.choose_action(state, explore=False, allowed_actions=(0, 1)), 1)
        self.assertEqual(agent.choose_action(state, explore=False, allowed_actions=(0,)), 0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            agent.choose_action(state, allowed_actions=())
        with self.assertRaisesRegex(ValueError, "must be in"):
            agent.choose_action(state, allowed_actions=(3,))

    def test_epsilon_decay_depends_on_decisions_not_training_updates(self):
        config = DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5)
        agent = DQNAgent(config, seed=3)
        state = np.zeros(20, dtype=np.float32)
        agent.choose_action(state, explore=True)
        self.assertEqual(agent.epsilon, 0.5)
        agent.update()
        self.assertEqual(agent.epsilon, 0.5)
        agent.choose_action(state, explore=False)
        self.assertEqual(agent.epsilon, 0.5)

    def test_batch_decisions_share_one_epsilon_snapshot(self):
        agent = DQNAgent(
            DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5),
            seed=17,
        )
        state = np.zeros(20, dtype=np.float32)
        agent.choose_action(state, explore=True, advance_exploration=False)
        agent.choose_action(state, explore=True, advance_exploration=False)
        self.assertEqual(agent.epsilon, 1.0)
        agent.advance_exploration(True, decisions=2)
        self.assertEqual(agent.epsilon, 0.25)

    def test_batched_q_values_match_individual_forward_passes(self):
        agent = DQNAgent(DQNConfig(), seed=19)
        states = np.asarray(
            [
                np.linspace(0.0, 1.0, 20, dtype=np.float32),
                np.linspace(1.0, 0.0, 20, dtype=np.float32),
                np.full(20, 0.25, dtype=np.float32),
            ]
        )
        batched = agent.q_values_batch(states)
        individual = np.stack([agent.q_values(state) for state in states])
        np.testing.assert_allclose(batched, individual, rtol=1e-6, atol=1e-7)

    def test_precomputed_q_action_respects_mask_and_tie_breaking(self):
        agent = DQNAgent(
            DQNConfig(epsilon_start=0.0, epsilon_end=0.0),
            seed=17,
        )
        q_values = np.asarray([5.0, 5.0, 100.0], dtype=np.float32)
        action = agent.choose_action_from_q(
            q_values,
            explore=False,
            allowed_actions=(0, 1),
        )
        self.assertEqual(action, 0)

    def test_stackelberg_action_can_guide_exploration(self):
        agent = DQNAgent(
            DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5), seed=17
        )
        state = np.zeros(20, dtype=np.float32)
        action = agent.choose_action(
            state,
            explore=True,
            allowed_actions=(0, 2),
            exploration_action=2,
        )
        self.assertEqual(action, 2)
        self.assertEqual(agent.epsilon, 0.5)
        with self.assertRaisesRegex(ValueError, "exploration_action"):
            agent.choose_action(state, allowed_actions=(0, 1), exploration_action=2)

    def test_residual_policy_can_override_game_prior(self):
        config = DQNConfig(epsilon_start=0.0, epsilon_end=0.0)
        agent = DQNAgent(config, seed=5)
        state = np.zeros(20, dtype=np.float32)
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 10.0]))
            agent.target.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 8.0]))
        action = agent.choose_residual_action(
            state,
            {0: 1.0, 2: 0.0},
            residual_weight=1.25,
            explore=False,
            allowed_actions=(0, 2),
        )
        self.assertEqual(action, 2)

    def test_residual_policy_requires_online_and_target_advantage(self):
        config = DQNConfig(epsilon_start=0.0, epsilon_end=0.0)
        agent = DQNAgent(config, seed=5)
        state = np.zeros(20, dtype=np.float32)
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, 10.0]))
            agent.target.layers[-1].bias.copy_(torch.tensor([0.0, 0.0, -10.0]))
        action = agent.choose_residual_action(
            state,
            {0: 1.0, 2: 0.0},
            residual_weight=1.0,
            explore=False,
            allowed_actions=(0, 2),
        )
        self.assertEqual(action, 0)

    def test_checkpoint_loads_policy_without_training_counters(self):
        config = DQNConfig()
        trained = DQNAgent(config, seed=3)
        trained.transition_count = 12
        trained.update_count = 4
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.pt"
            trained.save(path)
            loaded = DQNAgent(config, seed=9)
            loaded.load(path)
        self.assertEqual(loaded.transition_count, 0)
        self.assertEqual(loaded.update_count, 0)
        for expected, actual in zip(trained.online.parameters(), loaded.online.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
