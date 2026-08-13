from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .config import DQNConfig
from .domain import OffloadAction, OffloadEstimate, Task, VehicleState


class QNetwork(nn.Module):
    def __init__(self, state_size: int = 20, action_size: int = 3, hidden_sizes: tuple[int, int] = (256, 128)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


@dataclass(slots=True)
class ReplayTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    preferred_action: int = -1
    preference_weight: float = 0.0
    preference_action_mask: np.ndarray | None = None

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done
        yield self.next_action_mask

    def __getitem__(self, index: int):
        return tuple(self)[index]


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, sampling: str = "ring"):
        self.capacity = capacity
        self.sampling = sampling
        self._items: list[ReplayTransition] = []
        self._position = 0
        self._rng = random.Random(seed)
        self._strata: list[list[ReplayTransition]] = [
            [] for _ in range(4)
        ]
        self._stratum_seen = [0] * 4
        self._stratum_capacities = [0] * 4

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
        preferred_action: int = -1,
        preference_weight: float = 0.0,
        preference_action_mask: np.ndarray | None = None,
    ) -> None:
        item = ReplayTransition(
            state.copy(),
            action,
            reward,
            next_state.copy(),
            done,
            next_action_mask.copy(),
            preferred_action,
            preference_weight,
            (
                preference_action_mask.copy()
                if preference_action_mask is not None
                else np.ones(next_action_mask.shape, dtype=np.bool_)
            ),
        )
        if self.sampling == "load_stratified":
            stratum = self._load_stratum(state)
            first_in_stratum = self._stratum_seen[stratum] == 0
            self._stratum_seen[stratum] += 1
            active_strata = sum(seen > 0 for seen in self._stratum_seen)
            equal_share = self.capacity // active_strata
            if (
                first_in_stratum
                or (
                    self._stratum_capacities[stratum] < equal_share
                    and self._stratum_seen[stratum]
                    > self._stratum_capacities[stratum]
                )
            ):
                self._rebalance_stratum_capacities()
            bucket = self._strata[stratum]
            bucket_capacity = self._stratum_capacities[stratum]
            if len(bucket) < bucket_capacity:
                bucket.append(item)
            elif bucket_capacity:
                replacement = self._rng.randrange(self._stratum_seen[stratum])
                if replacement < bucket_capacity:
                    bucket[replacement] = item
            return
        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._position] = item
        self._position = (self._position + 1) % self.capacity

    def sample(
        self,
        size: int,
    ) -> list[ReplayTransition]:
        if self.sampling == "load_stratified":
            total = len(self)
            sampled = self._rng.sample(range(total), size)
            cumulative = []
            running = 0
            for bucket in self._strata:
                running += len(bucket)
                cumulative.append(running)
            result = []
            for logical_index in sampled:
                previous = 0
                for bucket, boundary in zip(self._strata, cumulative):
                    if logical_index < boundary:
                        result.append(bucket[logical_index - previous])
                        break
                    previous = boundary
            return result
        sampled = self._rng.sample(range(len(self._items)), size)
        if len(self._items) < self.capacity:
            return [self._items[index] for index in sampled]
        # _position is the oldest item once the ring is full. Mapping sampled
        # logical indices preserves the former deque's deterministic order
        # without allocating a 10,000-item list on every update.
        return [self._items[(self._position + index) % self.capacity] for index in sampled]

    def __len__(self) -> int:
        if self.sampling == "load_stratified":
            return sum(len(bucket) for bucket in self._strata)
        return len(self._items)

    @staticmethod
    def _load_stratum(state: np.ndarray) -> int:
        cloud_capacity_ratio = float(state[10])
        if cloud_capacity_ratio < 0.5:
            return 0
        if cloud_capacity_ratio < 0.85:
            return 1
        if cloud_capacity_ratio < 1.2:
            return 2
        return 3

    def _rebalance_stratum_capacities(self) -> None:
        """Share unused quota while retaining scarce congestion regimes."""
        capacities = [0] * len(self._strata)
        remaining = self.capacity
        unassigned = {
            index for index, seen in enumerate(self._stratum_seen) if seen > 0
        }
        while unassigned:
            if len(unassigned) == 1:
                capacities[unassigned.pop()] = remaining
                remaining = 0
                break
            share, remainder = divmod(remaining, len(unassigned))
            scarce = [
                index
                for index in sorted(unassigned)
                if self._stratum_seen[index] <= share
            ]
            if scarce:
                for index in scarce:
                    capacities[index] = self._stratum_seen[index]
                    remaining -= capacities[index]
                    unassigned.remove(index)
                continue
            for offset, index in enumerate(sorted(unassigned)):
                capacities[index] = share + int(offset < remainder)
            remaining = 0
            break
        self._stratum_capacities = capacities
        for bucket, bucket_capacity in zip(self._strata, capacities):
            while len(bucket) > bucket_capacity:
                replacement = self._rng.randrange(len(bucket))
                bucket[replacement] = bucket[-1]
                bucket.pop()


class DQNAgent:
    def __init__(self, config: DQNConfig, seed: int):
        self.config = config
        self.device = torch.device(config.device)
        torch.set_num_threads(config.intraop_threads)
        if torch.get_num_interop_threads() != config.interop_threads:
            try:
                torch.set_num_interop_threads(config.interop_threads)
            except RuntimeError:
                if torch.get_num_interop_threads() != config.interop_threads:
                    raise
        self._rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.online = QNetwork(config.state_size, config.action_size, config.hidden_sizes).to(self.device)
        self.target = QNetwork(config.state_size, config.action_size, config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(
            config.replay_capacity,
            seed,
            sampling=config.replay_sampling,
        )
        self.epsilon = config.epsilon_start
        self.transition_count = 0
        self.update_count = 0
        self.last_loss: float | None = None

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "epsilon": self.epsilon,
                "transition_count": self.transition_count,
                "update_count": self.update_count,
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        self.epsilon = self.config.epsilon_end

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        allowed_actions: Iterable[int] | None = None,
        exploration_action: int | None = None,
        advance_exploration: bool = True,
    ) -> int:
        allowed = self._normalize_allowed_actions(allowed_actions)
        if exploration_action is not None and exploration_action not in allowed:
            raise ValueError("exploration_action must be allowed")
        q_values = None
        if len(allowed) > 1 and not (
            explore and self._rng.random() < self.epsilon
        ):
            q_values = self.q_values(state)
        return self.choose_action_from_q(
            q_values,
            explore=explore,
            allowed_actions=allowed,
            exploration_action=exploration_action,
            advance_exploration=advance_exploration,
            exploration_checked=True,
        )

    def choose_action_from_q(
        self,
        q_values: np.ndarray | None,
        explore: bool = True,
        allowed_actions: Iterable[int] | None = None,
        exploration_action: int | None = None,
        advance_exploration: bool = True,
        exploration_checked: bool = False,
    ) -> int:
        allowed = self._normalize_allowed_actions(allowed_actions)
        if exploration_action is not None and exploration_action not in allowed:
            raise ValueError("exploration_action must be allowed")
        if len(allowed) == 1:
            action = allowed[0]
        elif (
            explore
            and (
                (
                    not exploration_checked
                    and self._rng.random() < self.epsilon
                )
                or (exploration_checked and q_values is None)
            )
        ):
            action = exploration_action if exploration_action is not None else self._rng.choice(allowed)
        else:
            if q_values is None:
                raise ValueError("q_values are required for a non-exploratory decision")
            action = max(allowed, key=lambda candidate: (float(q_values[candidate]), -candidate))
        self.advance_exploration(explore and advance_exploration)
        return action

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_values_batch(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]

    def q_values_batch(
        self,
        states: np.ndarray,
        target: bool = False,
    ) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.state_size:
            raise ValueError(
                f"states must have shape (N, {self.config.state_size})"
            )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        network = self.target if target else self.online
        with torch.inference_mode():
            return network(tensor).cpu().numpy().copy()

    def choose_residual_action(
        self,
        state: np.ndarray,
        immediate_rewards: dict[int, float],
        residual_weight: float,
        explore: bool = True,
        allowed_actions: Iterable[int] | None = None,
        advance_exploration: bool = True,
    ) -> int:
        """Blend immediate private reward with a conservative learned advantage."""
        allowed = self._normalize_allowed_actions(allowed_actions)
        online_q = None
        target_q = None
        exploration_checked = False
        if len(allowed) > 1:
            explore_now = explore and self._rng.random() < self.epsilon
            exploration_checked = True
            if not explore_now:
                online_q = self.q_values(state)
                target_q = self.q_values_batch(
                    np.asarray(state, dtype=np.float32).reshape(1, -1),
                    target=True,
                )[0]
        return self.choose_residual_action_from_q(
            online_q,
            target_q,
            immediate_rewards,
            residual_weight,
            explore=explore,
            allowed_actions=allowed,
            advance_exploration=advance_exploration,
            exploration_checked=exploration_checked,
        )

    def choose_residual_action_from_q(
        self,
        online_q: np.ndarray | None,
        target_q: np.ndarray | None,
        immediate_rewards: dict[int, float],
        residual_weight: float,
        explore: bool = True,
        allowed_actions: Iterable[int] | None = None,
        advance_exploration: bool = True,
        exploration_checked: bool = False,
    ) -> int:
        allowed = self._normalize_allowed_actions(allowed_actions)
        if len(allowed) == 1:
            action = allowed[0]
        elif (
            explore
            and (
                (
                    not exploration_checked
                    and self._rng.random() < self.epsilon
                )
                or (exploration_checked and online_q is None)
            )
        ):
            action = self._rng.choice(allowed)
        else:
            if online_q is None or target_q is None:
                raise ValueError(
                    "online and target Q values are required for a residual decision"
                )
            baseline_action = max(
                allowed,
                key=lambda action: (immediate_rewards[action], -action),
            )
            immediate_advantage = np.asarray(
                [
                    immediate_rewards[action] - immediate_rewards[baseline_action]
                    for action in allowed
                ],
                dtype=np.float64,
            )
            online_advantage = np.asarray(
                [online_q[action] - online_q[baseline_action] for action in allowed],
                dtype=np.float64,
            )
            target_advantage = np.asarray(
                [target_q[action] - target_q[baseline_action] for action in allowed],
                dtype=np.float64,
            )
            learned_advantage = np.minimum(online_advantage, target_advantage)
            weight = min(max(residual_weight, 0.0), 1.0)
            combined = (
                (1.0 - weight) * immediate_advantage
                + weight * learned_advantage
            )
            best = max(range(len(allowed)), key=lambda index: (combined[index], -allowed[index]))
            action = allowed[best]
        self.advance_exploration(explore and advance_exploration)
        return action

    def advance_exploration(self, explore: bool = True, decisions: int = 1) -> None:
        """Advance the policy schedule once for one environment decision."""
        if decisions < 0:
            raise ValueError("decisions must be non-negative")
        if explore and decisions:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay**decisions,
            )

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_allowed_actions: Iterable[int] | None = None,
        preferred_action: int = -1,
        preference_weight: float = 0.0,
        preferred_allowed_actions: Iterable[int] | None = None,
    ) -> None:
        mask = np.ones(self.config.action_size, dtype=np.bool_)
        if next_allowed_actions is not None:
            allowed = self._normalize_allowed_actions(next_allowed_actions)
            mask[:] = False
            mask[list(allowed)] = True
        if preferred_action < -1 or preferred_action >= self.config.action_size:
            raise ValueError("preferred_action is outside the action space")
        preference_mask = np.ones(self.config.action_size, dtype=np.bool_)
        if preferred_allowed_actions is not None:
            preference_allowed = self._normalize_allowed_actions(
                preferred_allowed_actions
            )
            preference_mask[:] = False
            preference_mask[list(preference_allowed)] = True
        self.replay.add(
            state,
            action,
            reward,
            next_state,
            done,
            mask,
            preferred_action,
            max(0.0, float(preference_weight)),
            preference_mask,
        )
        self.transition_count += 1

    def update(self) -> float | None:
        required = max(self.config.batch_size, self.config.warmup_transitions)
        if len(self.replay) < required:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        preferred_actions = [item.preferred_action for item in batch]
        preference_weights = [item.preference_weight for item in batch]
        preference_masks = [
            item.preference_action_mask for item in batch
        ]
        state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask_tensor = torch.as_tensor(np.stack(next_masks), dtype=torch.bool, device=self.device)

        all_q_values = self.online(state_tensor)
        q_values = all_q_values.gather(1, action_tensor)
        with torch.no_grad():
            # Double DQN: the online network selects the next feasible action;
            # the target network evaluates it.
            online_next = self.online(next_tensor).masked_fill(~next_mask_tensor, float("-inf"))
            next_actions = online_next.argmax(dim=1, keepdim=True)
            next_q = self.target(next_tensor).gather(1, next_actions)
            targets = reward_tensor + self.config.gamma * next_q * (1.0 - done_tensor)
        td_loss = nn.functional.smooth_l1_loss(
            q_values,
            targets,
            beta=self.config.huber_delta,
        )
        guidance_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.config.game_guidance_weight > 0:
            preferred_tensor = torch.as_tensor(
                preferred_actions,
                dtype=torch.int64,
                device=self.device,
            )
            guidance_weights = torch.as_tensor(
                preference_weights,
                dtype=torch.float32,
                device=self.device,
            )
            preference_mask_tensor = torch.as_tensor(
                np.stack(preference_masks),
                dtype=torch.bool,
                device=self.device,
            )
            preferred_valid = preferred_tensor.clamp_min(0)
            preferred_is_allowed = preference_mask_tensor.gather(
                1,
                preferred_valid.unsqueeze(1),
            ).squeeze(1)
            rival_exists = (
                preference_mask_tensor.sum(dim=1) > 1
            )
            valid = (
                (preferred_tensor >= 0)
                & (guidance_weights > 0)
                & preferred_is_allowed
                & rival_exists
            )
            if valid.any():
                valid_q = all_q_values[valid]
                valid_preferred = preferred_tensor[valid]
                preferred_q = valid_q.gather(
                    1,
                    valid_preferred.unsqueeze(1),
                ).squeeze(1)
                rival_mask = torch.nn.functional.one_hot(
                    valid_preferred,
                    num_classes=self.config.action_size,
                ).bool()
                allowed_rivals = (
                    preference_mask_tensor[valid] & ~rival_mask
                )
                rival_q = valid_q.masked_fill(
                    ~allowed_rivals,
                    float("-inf"),
                ).max(dim=1).values
                ranking_error = torch.relu(
                    self.config.game_guidance_margin - preferred_q + rival_q
                )
                valid_weights = guidance_weights[valid]
                guidance_loss = (
                    ranking_error * valid_weights
                ).sum() / valid_weights.sum().clamp_min(1e-12)
        loss = td_loss + self.config.game_guidance_weight * guidance_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.config.gradient_clip_norm)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def _normalize_allowed_actions(self, allowed_actions: Iterable[int] | None) -> tuple[int, ...]:
        allowed = (
            tuple(range(self.config.action_size))
            if allowed_actions is None
            else tuple(sorted(set(int(action) for action in allowed_actions)))
        )
        if not allowed:
            raise ValueError("DQN requires at least one allowed action")
        if allowed[0] < 0 or allowed[-1] >= self.config.action_size:
            raise ValueError(f"allowed actions must be in [0, {self.config.action_size - 1}]")
        return allowed


def build_state(
    task: Task,
    vehicle: VehicleState,
    current_price: float,
    neighbors: int,
    cloud_queue_ratio: float,
    cloud_capacity_ratio: float,
    v2v_target_workload_ratio: float,
    candidates: dict[OffloadAction, OffloadEstimate],
    compute_max_cycles: float,
    data_max_mb: float,
    deadline_max_s: float,
    energy_scale_j: float,
    payment_scale: float,
    cloud_max_price: float,
) -> np.ndarray:
    def ratio(value: float, scale: float, limit: float = 2.0) -> float:
        if not np.isfinite(value):
            return limit
        return min(max(value / max(scale, 1e-12), 0.0), limit)

    ordered = [candidates[action] for action in OffloadAction]
    values = [
        ratio(task.compute_cycles, compute_max_cycles),
        ratio(task.data_size_mb, data_max_mb),
        ratio(task.deadline_s, deadline_max_s),
        min(max(task.urgency, 0.0), 1.0),
        min(max(vehicle.energy_level, 0.0), 1.0),
        min(max(vehicle.speed_mps / 40.0, 0.0), 1.0),
        min(neighbors / 10.0, 1.0),
        min(max(v2v_target_workload_ratio, 0.0), 2.0),
        ratio(current_price, cloud_max_price),
        min(max(cloud_queue_ratio, 0.0), 2.0),
        min(max(cloud_capacity_ratio, 0.0), 2.0),
        *[ratio(candidate.delay_s, task.deadline_s) for candidate in ordered],
        *[ratio(candidate.energy_j, energy_scale_j) for candidate in ordered],
        *[
            ratio(candidates[action].payment, payment_scale)
            for action in (OffloadAction.V2V, OffloadAction.V2I)
        ],
        min(max(vehicle.workload_cycles / max(vehicle.compute_hz, 1.0), 0.0), 2.0),
    ]
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (20,):
        raise AssertionError(f"expected 20 state values, received {state.shape}")
    return state
