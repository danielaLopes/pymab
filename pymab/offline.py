"""Offline policy evaluation for logged bandit feedback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from pymab.policies.policy import ContextualPolicy, FloatArray, Policy


@dataclass(frozen=True)
class OfflineReplayResult:
    """Result of replay evaluation on logged bandit data."""

    n_logged_events: int
    n_accepted_events: int
    selected_actions: np.ndarray
    accepted_actions: np.ndarray
    accepted_rewards: FloatArray
    cumulative_rewards: FloatArray
    ips_estimate: float | None = None

    @property
    def acceptance_rate(self) -> float:
        """Fraction of logged events accepted by replay matching."""

        if self.n_logged_events == 0:
            return 0.0
        return self.n_accepted_events / self.n_logged_events

    @property
    def average_reward(self) -> float:
        """Mean reward over accepted replay events."""

        if self.n_accepted_events == 0:
            return 0.0
        return float(np.mean(self.accepted_rewards))

    @property
    def total_reward(self) -> float:
        """Total reward over accepted replay events."""

        return float(np.sum(self.accepted_rewards))


def replay_evaluate(
    policy: Policy | ContextualPolicy,
    *,
    logged_actions: Sequence[int] | np.ndarray,
    logged_rewards: Sequence[float] | np.ndarray,
    contexts: np.ndarray | None = None,
    propensities: Sequence[float] | np.ndarray | None = None,
    seed: int | None = None,
    clone_policy: bool = True,
) -> OfflineReplayResult:
    """Evaluate a policy from logged bandit feedback using replay matching.

    The policy is run against each logged event. When the policy chooses the
    same action as the logging policy, the event is accepted, the reward is
    observed, and the policy is updated. If logging propensities are supplied,
    the result also includes a simple inverse-propensity estimate over the full
    log.
    """

    actions = np.asarray(logged_actions, dtype=int)
    rewards = np.asarray(logged_rewards, dtype=float)
    if actions.ndim != 1 or rewards.ndim != 1:
        raise ValueError("logged_actions and logged_rewards must be 1D")
    if actions.shape[0] != rewards.shape[0]:
        raise ValueError("logged_actions and logged_rewards must have equal length")
    if actions.size == 0:
        raise ValueError("logged data must contain at least one event")

    propensity_values: FloatArray | None = None
    if propensities is not None:
        propensity_values = np.asarray(propensities, dtype=float)
        if propensity_values.shape != rewards.shape:
            raise ValueError("propensities must match logged_rewards shape")
        if np.any(propensity_values <= 0):
            raise ValueError("propensities must be positive")

    context_values: np.ndarray | None = None
    if contexts is not None:
        context_values = np.asarray(contexts, dtype=float)
        if context_values.shape[0] != actions.size:
            raise ValueError("contexts must have one row per logged event")

    evaluator_policy = policy.clone() if clone_policy else policy
    evaluator_policy.reset()
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    accepted_actions: list[int] = []
    accepted_rewards: list[float] = []
    ips_total = 0.0

    for index, logged_action in enumerate(actions):
        if isinstance(evaluator_policy, ContextualPolicy):
            if context_values is None:
                raise ValueError("contexts are required for contextual policies")
            context = cast(FloatArray, context_values[index])
            selected_action = evaluator_policy.select_action(context=context, rng=rng)
        else:
            selected_action = evaluator_policy.select_action(rng=rng)
        selected.append(selected_action)

        if selected_action != int(logged_action):
            continue

        reward = float(rewards[index])
        if isinstance(evaluator_policy, ContextualPolicy):
            if context_values is None:
                raise ValueError("contexts are required for contextual policies")
            evaluator_policy.update(
                action=selected_action,
                reward=reward,
                context=cast(FloatArray, context_values[index]),
            )
        else:
            evaluator_policy.update(action=selected_action, reward=reward)
        accepted_actions.append(selected_action)
        accepted_rewards.append(reward)
        if propensity_values is not None:
            ips_total += reward / float(propensity_values[index])

    accepted_reward_array = np.array(accepted_rewards, dtype=float)
    return OfflineReplayResult(
        n_logged_events=int(actions.size),
        n_accepted_events=len(accepted_rewards),
        selected_actions=np.array(selected, dtype=int),
        accepted_actions=np.array(accepted_actions, dtype=int),
        accepted_rewards=accepted_reward_array,
        cumulative_rewards=np.cumsum(accepted_reward_array),
        ips_estimate=(
            None if propensity_values is None else float(ips_total / actions.size)
        ),
    )


__all__ = ["OfflineReplayResult", "replay_evaluate"]
