"""Experiment orchestration for PyMAB."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from pymab.environments import BanditEnvironment, LinearContextualEnvironment
from pymab.policies.policy import ContextualPolicy, Policy


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode simulation traces."""

    rewards: np.ndarray
    actions: np.ndarray
    expected_rewards: np.ndarray
    optimal_actions: np.ndarray
    optimal_values: np.ndarray
    q_values: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Immutable result arrays from a bandit experiment."""

    rewards: np.ndarray
    actions: np.ndarray
    expected_rewards: np.ndarray
    optimal_actions: np.ndarray
    optimal_values: np.ndarray
    policy_names: tuple[str, ...]
    q_values: np.ndarray | None = None

    @property
    def n_episodes(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def n_policies(self) -> int:
        return int(self.rewards.shape[2])

    @property
    def regret(self) -> np.ndarray:
        return cast(
            np.ndarray, self.optimal_values[:, :, np.newaxis] - self.expected_rewards
        )

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(np.mean(self.regret, axis=0), axis=0)

    @property
    def average_reward_by_step(self) -> np.ndarray:
        return np.asarray(np.mean(self.rewards, axis=0))

    @property
    def cumulative_reward_by_step(self) -> np.ndarray:
        return np.cumsum(self.average_reward_by_step, axis=0)

    @property
    def optimal_action_rate_by_step(self) -> np.ndarray:
        optimal = self.actions == self.optimal_actions[:, :, np.newaxis]
        return cast(np.ndarray, np.mean(optimal, axis=0))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the simulation result."""

        return {
            "rewards": self.rewards.tolist(),
            "actions": self.actions.tolist(),
            "expected_rewards": self.expected_rewards.tolist(),
            "optimal_actions": self.optimal_actions.tolist(),
            "optimal_values": self.optimal_values.tolist(),
            "policy_names": list(self.policy_names),
            "q_values": None if self.q_values is None else self.q_values.tolist(),
        }

    def to_pandas(self) -> Any:
        """Return tidy per-step results as a pandas DataFrame.

        Install ``pymab[analysis]`` to make pandas available.
        """

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pymab[analysis] to use to_pandas().") from exc

        rows: list[dict[str, Any]] = []
        for episode in range(self.n_episodes):
            for step in range(self.n_steps):
                optimal_action = int(self.optimal_actions[episode, step])
                optimal_value = float(self.optimal_values[episode, step])
                for policy_index, policy_name in enumerate(self.policy_names):
                    expected_reward = float(
                        self.expected_rewards[episode, step, policy_index]
                    )
                    rows.append(
                        {
                            "episode": episode,
                            "step": step,
                            "policy_index": policy_index,
                            "policy_name": policy_name,
                            "action": int(self.actions[episode, step, policy_index]),
                            "reward": float(self.rewards[episode, step, policy_index]),
                            "expected_reward": expected_reward,
                            "optimal_action": optimal_action,
                            "optimal_value": optimal_value,
                            "regret": optimal_value - expected_reward,
                            "selected_optimal_action": bool(
                                self.actions[episode, step, policy_index]
                                == optimal_action
                            ),
                        }
                    )
        return pd.DataFrame.from_records(rows)

    def save_npz(self, path: str | Path) -> None:
        """Persist result arrays and metadata to a compressed NumPy archive."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            rewards=self.rewards,
            actions=self.actions,
            expected_rewards=self.expected_rewards,
            optimal_actions=self.optimal_actions,
            optimal_values=self.optimal_values,
            q_values=np.array([]) if self.q_values is None else self.q_values,
            has_q_values=np.array(self.q_values is not None),
            policy_names=np.array(self.policy_names, dtype=str),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> SimulationResult:
        """Load a result previously saved with :meth:`save_npz`."""

        with np.load(Path(path)) as data:
            has_q_values = bool(data["has_q_values"].item())
            return cls(
                rewards=np.asarray(data["rewards"], dtype=float),
                actions=np.asarray(data["actions"], dtype=int),
                expected_rewards=np.asarray(data["expected_rewards"], dtype=float),
                optimal_actions=np.asarray(data["optimal_actions"], dtype=int),
                optimal_values=np.asarray(data["optimal_values"], dtype=float),
                policy_names=tuple(str(name) for name in data["policy_names"]),
                q_values=(
                    np.asarray(data["q_values"], dtype=float) if has_q_values else None
                ),
            )

    def save_json(self, path: str | Path) -> None:
        """Persist a JSON representation for lightweight inspection."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a repeated bandit experiment."""

    n_episodes: int
    n_steps: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_episodes <= 0:
            raise ValueError("n_episodes must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")


class Experiment:
    """Run policies against an environment."""

    def __init__(
        self,
        *,
        environment: BanditEnvironment | LinearContextualEnvironment,
        policies: Sequence[Policy | ContextualPolicy],
        config: ExperimentConfig,
    ) -> None:
        if not policies:
            raise ValueError("at least one policy is required")
        self.environment = environment
        self.policies = tuple(policies)
        self.config = config
        self._validate_policy_arms()

    def run(self) -> SimulationResult:
        """Run the configured experiment."""

        if isinstance(self.environment, LinearContextualEnvironment):
            return self._run_contextual()
        return self._run_standard()

    def _run_standard(self) -> SimulationResult:
        config = self.config
        base_environment = self.environment
        if not isinstance(base_environment, BanditEnvironment):
            raise TypeError("standard environments require non-contextual policies")
        n_policies = len(self.policies)
        rewards = np.zeros((config.n_episodes, config.n_steps, n_policies))
        actions = np.zeros((config.n_episodes, config.n_steps, n_policies), dtype=int)
        expected = np.zeros_like(rewards)
        optimal_actions = np.zeros((config.n_episodes, config.n_steps), dtype=int)
        optimal_values = np.zeros((config.n_episodes, config.n_steps))
        q_values = np.zeros(
            (config.n_episodes, config.n_steps, base_environment.n_arms)
        )
        seed_sequence = np.random.SeedSequence(config.seed)

        for episode, child_seed in enumerate(seed_sequence.spawn(config.n_episodes)):
            rng = np.random.default_rng(child_seed)
            environment = base_environment.copy()
            policies = self._fresh_policies()
            for step in range(config.n_steps):
                environment.advance(step=step, rng=rng)
                q_values[episode, step] = environment.q_values
                optimal_actions[episode, step] = environment.optimal_action
                optimal_values[episode, step] = environment.optimal_value
                for policy_index, policy in enumerate(policies):
                    if not isinstance(policy, Policy):
                        raise TypeError(
                            "standard environments require non-contextual policies"
                        )
                    action = policy.select_action(rng=rng)
                    reward = environment.step(action, rng=rng)
                    policy.update(action=action, reward=reward)
                    actions[episode, step, policy_index] = action
                    rewards[episode, step, policy_index] = reward
                    expected[episode, step, policy_index] = environment.expected_reward(
                        action
                    )

        return SimulationResult(
            rewards=rewards,
            actions=actions,
            expected_rewards=expected,
            optimal_actions=optimal_actions,
            optimal_values=optimal_values,
            policy_names=tuple(repr(policy) for policy in self.policies),
            q_values=q_values,
        )

    def _run_contextual(self) -> SimulationResult:
        config = self.config
        environment = self.environment
        if not isinstance(environment, LinearContextualEnvironment):
            raise TypeError("contextual runs require a contextual environment")
        n_policies = len(self.policies)
        rewards = np.zeros((config.n_episodes, config.n_steps, n_policies))
        actions = np.zeros((config.n_episodes, config.n_steps, n_policies), dtype=int)
        expected = np.zeros_like(rewards)
        optimal_actions = np.zeros((config.n_episodes, config.n_steps), dtype=int)
        optimal_values = np.zeros((config.n_episodes, config.n_steps))
        seed_sequence = np.random.SeedSequence(config.seed)

        for episode, child_seed in enumerate(seed_sequence.spawn(config.n_episodes)):
            rng = np.random.default_rng(child_seed)
            episode_environment = environment.copy()
            policies = self._fresh_policies()
            for step in range(config.n_steps):
                context = episode_environment.context(rng)
                true_rewards = episode_environment.expected_rewards(context)
                optimal_actions[episode, step] = int(np.argmax(true_rewards))
                optimal_values[episode, step] = float(np.max(true_rewards))
                for policy_index, policy in enumerate(policies):
                    if not isinstance(policy, ContextualPolicy):
                        raise TypeError(
                            "contextual environments require contextual policies"
                        )
                    action = policy.select_action(context=context, rng=rng)
                    reward = episode_environment.step(action, context=context, rng=rng)
                    policy.update(action=action, reward=reward, context=context)
                    actions[episode, step, policy_index] = action
                    rewards[episode, step, policy_index] = reward
                    expected[episode, step, policy_index] = true_rewards[action]

        return SimulationResult(
            rewards=rewards,
            actions=actions,
            expected_rewards=expected,
            optimal_actions=optimal_actions,
            optimal_values=optimal_values,
            policy_names=tuple(repr(policy) for policy in self.policies),
        )

    def _fresh_policies(self) -> tuple[Policy | ContextualPolicy, ...]:
        fresh = tuple(policy.clone() for policy in self.policies)
        for policy in fresh:
            policy.reset()
        return fresh

    def _validate_policy_arms(self) -> None:
        expected = self.environment.n_arms
        for policy in self.policies:
            if policy.n_arms != expected:
                raise ValueError("all policies must match environment n_arms")
