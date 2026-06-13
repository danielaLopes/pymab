"""Upper-confidence-bound policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np

from pymab.policies.policy import ActionValuePolicy, FloatArray, choose_argmax


class UCBPolicy(ActionValuePolicy):
    """UCB1 policy for stationary rewards."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        c: float = 2.0,
        **_: object,
    ) -> None:
        if c <= 0:
            raise ValueError("c must be positive")
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        init = optimistic_initialization if initial_value is None else initial_value
        super().__init__(n_arms=int(arms), initial_value=float(init))
        self.c = float(c)

    def select_action(self, *, rng: np.random.Generator) -> int:
        unseen = np.flatnonzero(self.counts == 0)
        if unseen.size:
            return int(unseen[0])
        values = self.estimates + self._confidence_bonus()
        return choose_argmax(values, rng)

    def _confidence_bonus(self) -> FloatArray:
        return cast(FloatArray, np.sqrt(self.c * np.log(self.step + 1) / self.counts))

    def __repr__(self) -> str:
        return f"UCBPolicy(c={self.c}, initial_value={self.initial_value})"


class StationaryUCBPolicy(UCBPolicy):
    """Backward-compatible name for :class:`UCBPolicy`."""


class SlidingWindowUCBPolicy(UCBPolicy):
    """UCB with per-arm sliding-window value estimates."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        c: float = 2.0,
        window_size: int = 100,
        **kwargs: object,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self._windows: list[deque[float]]
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            initial_value=initial_value,
            optimistic_initialization=optimistic_initialization,
            c=c,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self._windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    def _update_estimate(self, *, action: int, reward: float) -> None:
        self._windows[action].append(float(reward))
        self.estimates[action] = float(np.mean(self._windows[action]))

    def _confidence_bonus(self) -> FloatArray:
        effective_counts = np.array(
            [max(len(window), 1) for window in self._windows], dtype=float
        )
        horizon = max(min(self.step + 1, self.window_size), 2)
        return cast(FloatArray, np.sqrt(self.c * np.log(horizon) / effective_counts))

    def __repr__(self) -> str:
        return f"SlidingWindowUCBPolicy(c={self.c}, window_size={self.window_size})"


class DiscountedUCBPolicy(UCBPolicy):
    """UCB with exponentially discounted counts and estimates."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        c: float = 2.0,
        discount_factor: float = 0.9,
        **kwargs: object,
    ) -> None:
        if not 0 < discount_factor < 1:
            raise ValueError("discount_factor must be in (0, 1)")
        self.discount_factor = float(discount_factor)
        self.discounted_counts: FloatArray
        self.discounted_sums: FloatArray
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            initial_value=initial_value,
            optimistic_initialization=optimistic_initialization,
            c=c,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self.discounted_counts = np.zeros(self.n_arms, dtype=float)
        self.discounted_sums = np.zeros(self.n_arms, dtype=float)

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
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
        return cast(FloatArray, np.sqrt(self.c * np.log(effective_horizon) / counts))

    def __repr__(self) -> str:
        return (
            f"DiscountedUCBPolicy(c={self.c}, discount_factor={self.discount_factor})"
        )


@dataclass
class UCBStats:
    """Diagnostic UCB state."""

    estimates: FloatArray
    counts: FloatArray
    bonuses: FloatArray
