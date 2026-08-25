"""Policies for non-stationary reward processes."""

from __future__ import annotations

from collections import deque
from typing import cast

import numpy as np

from pymab._reference.policies.thompson_sampling import BernoulliThompsonSamplingPolicy
from pymab._reference.policies.ucb import UCBPolicy
from pymab.policies.policy import validate_positive_integer
from pymab.types import FloatArray


class SlidingWindowUCBPolicy(UCBPolicy):
    """UCB over observations from the most recent global time steps."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        c: float = 2.0,
        reward_scale: float = 1.0,
        window_size: int = 100,
    ) -> None:
        validate_positive_integer(window_size, name="window_size")
        self.window_size = int(window_size)
        self._history: deque[tuple[int, int, float]]
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
        )

    def reset(self) -> None:
        super().reset()
        self._history = deque(maxlen=self.window_size)

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.step += 1
        self.total_reward += float(reward)
        self._history.append((self.step, int(action), float(reward)))
        self._refresh_window_state()

    def _refresh_window_state(self) -> None:
        self.counts.fill(0.0)
        sums = np.zeros(self.n_arms, dtype=float)
        for _, action, reward in self._history:
            self.counts[action] += 1.0
            sums[action] += reward
        observed = self.counts > 0
        self.estimates.fill(self.initial_value)
        self.estimates[observed] = sums[observed] / self.counts[observed]

    def _confidence_bonus(self) -> FloatArray:
        effective_counts = np.maximum(self.counts, 1.0)
        horizon = max(min(self.step, self.window_size), 2)
        return cast(
            FloatArray,
            self.reward_scale * np.sqrt(self.c * np.log(horizon) / effective_counts),
        )

    def __repr__(self) -> str:
        return (
            f"SlidingWindowUCBPolicy(c={self.c}, reward_scale={self.reward_scale}, "
            f"window_size={self.window_size})"
        )


class DiscountedUCBPolicy(UCBPolicy):
    """UCB with exponentially discounted counts and estimates."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        c: float = 2.0,
        reward_scale: float = 1.0,
        discount_factor: float = 0.9,
    ) -> None:
        if not np.isfinite(discount_factor) or not 0 < discount_factor < 1:
            raise ValueError("discount_factor must be in (0, 1)")
        self.discount_factor = float(discount_factor)
        self.discounted_counts: FloatArray
        self.discounted_sums: FloatArray
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
        )

    def reset(self) -> None:
        super().reset()
        self.discounted_counts = np.zeros(self.n_arms, dtype=float)
        self.discounted_sums = np.zeros(self.n_arms, dtype=float)

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.step += 1
        self.total_reward += float(reward)
        self.counts[action] += 1.0
        self.discounted_counts *= self.discount_factor
        self.discounted_sums *= self.discount_factor
        self.discounted_counts[action] += 1.0
        self.discounted_sums[action] += float(reward)
        observed = self.discounted_counts > 0
        self.estimates[observed] = (
            self.discounted_sums[observed] / self.discounted_counts[observed]
        )

    def _confidence_bonus(self) -> FloatArray:
        counts = np.maximum(self.discounted_counts, 1e-12)
        effective_horizon = max(float(np.sum(self.discounted_counts)), 2.0)
        return cast(
            FloatArray,
            self.reward_scale * np.sqrt(self.c * np.log(effective_horizon) / counts),
        )

    def __repr__(self) -> str:
        return (
            f"DiscountedUCBPolicy(c={self.c}, reward_scale={self.reward_scale}, "
            f"discount_factor={self.discount_factor})"
        )


class SlidingWindowBernoulliThompsonSamplingPolicy(BernoulliThompsonSamplingPolicy):
    """Beta-Bernoulli sampling over the most recent global time steps."""

    def __init__(
        self,
        *,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        window_size: int = 100,
    ) -> None:
        validate_positive_integer(window_size, name="window_size")
        self.window_size = int(window_size)
        self._history: deque[tuple[int, int, float]]
        super().__init__(
            n_arms=n_arms,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )

    def reset(self) -> None:
        super().reset()
        self._history = deque(maxlen=self.window_size)

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Sliding-window Thompson Sampling requires binary rewards")
        self._validate_action(action)
        self.step += 1
        self.total_reward += float(reward)
        self._history.append((self.step, int(action), float(reward)))
        self._refresh_posterior_from_history()

    def _refresh_posterior_from_history(self) -> None:
        self.counts.fill(0.0)
        self.successes.fill(0.0)
        self.failures.fill(0.0)
        self.estimates.fill(self.initial_value)
        for _, action, reward in self._history:
            self.counts[action] += 1.0
            self.successes[action] += reward
            self.failures[action] += 1.0 - reward
        observed = self.counts > 0
        self.estimates[observed] = self.successes[observed] / self.counts[observed]

    def __repr__(self) -> str:
        return (
            "SlidingWindowBernoulliThompsonSamplingPolicy("
            f"window_size={self.window_size}, alpha_prior={self.alpha_prior}, "
            f"beta_prior={self.beta_prior})"
        )


class DiscountedBernoulliThompsonSamplingPolicy(BernoulliThompsonSamplingPolicy):
    """Beta-Bernoulli Thompson Sampling with exponential forgetting."""

    def __init__(
        self,
        *,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        discount_factor: float = 0.95,
    ) -> None:
        if not np.isfinite(discount_factor) or not 0 < discount_factor < 1:
            raise ValueError("discount_factor must be in (0, 1)")
        self.discount_factor = float(discount_factor)
        super().__init__(
            n_arms=n_arms,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Discounted Thompson Sampling requires binary rewards")
        self._validate_action(action)
        value = float(reward)
        self.step += 1
        self.total_reward += value
        self.counts *= self.discount_factor
        self.successes *= self.discount_factor
        self.failures *= self.discount_factor
        self.counts[action] += 1.0
        if value > 0:
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0
        observed = self.counts > 0
        self.estimates[observed] = self.successes[observed] / self.counts[observed]

    def __repr__(self) -> str:
        return (
            "DiscountedBernoulliThompsonSamplingPolicy("
            f"discount_factor={self.discount_factor}, "
            f"alpha_prior={self.alpha_prior}, beta_prior={self.beta_prior})"
        )


__all__ = [
    "DiscountedBernoulliThompsonSamplingPolicy",
    "DiscountedUCBPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
]
