"""Stationary upper-confidence-bound policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from pymab.policies.policy import (
    ActionValuePolicy,
    choose_argmax,
    validate_positive,
    validate_positive_integer,
)
from pymab.types import FloatArray, PolicyCapabilities, RewardDomain


class UCBPolicy(ActionValuePolicy):
    """UCB1 policy for stationary sub-Gaussian rewards."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        c: float = 2.0,
        reward_scale: float = 1.0,
    ) -> None:
        validate_positive(c, name="c")
        validate_positive(reward_scale, name="reward_scale")
        super().__init__(n_arms=n_arms, initial_value=float(initial_value))
        self.c = float(c)
        self.reward_scale = float(reward_scale)

    def select_action(self, *, rng: np.random.Generator) -> int:
        unseen = np.flatnonzero(self.counts == 0)
        if unseen.size:
            return int(unseen[0])
        values = self.estimates + self._confidence_bonus()
        return choose_argmax(values, rng)

    def _confidence_bonus(self) -> FloatArray:
        return cast(
            FloatArray,
            self.reward_scale * np.sqrt(self.c * np.log(self.step + 1) / self.counts),
        )

    def __repr__(self) -> str:
        return (
            f"UCBPolicy(c={self.c}, reward_scale={self.reward_scale}, "
            f"initial_value={self.initial_value})"
        )


class KLUCBPolicy(UCBPolicy):
    """KL-UCB for Bernoulli rewards."""

    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=frozenset({RewardDomain.BINARY}),
    )

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        c: float = 3.0,
        tolerance: float = 1e-6,
        max_iterations: int = 32,
    ) -> None:
        validate_positive(tolerance, name="tolerance")
        validate_positive_integer(max_iterations, name="max_iterations")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            reward_scale=1.0,
        )

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("KL-UCB requires binary rewards")
        super().update(action=action, reward=float(reward))

    def select_action(self, *, rng: np.random.Generator) -> int:
        unseen = np.flatnonzero(self.counts == 0)
        if unseen.size:
            return int(unseen[0])
        return choose_argmax(self.indices(), rng)

    def indices(self) -> FloatArray:
        """Return current KL-UCB indices for all arms."""

        budget = np.log(max(self.step, 2)) + self.c * np.log(
            max(np.log(max(self.step, 3)), 1.0)
        )
        values = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            values[arm] = self._solve_index(
                mean=float(np.clip(self.estimates[arm], 0.0, 1.0)),
                budget=float(budget / self.counts[arm]),
            )
        return values

    def _solve_index(self, *, mean: float, budget: float) -> float:
        if mean >= 1.0:
            return 1.0
        low = mean
        high = 1.0
        for _ in range(self.max_iterations):
            midpoint = (low + high) / 2.0
            divergence = _bernoulli_kl(mean, midpoint)
            if divergence <= budget:
                low = midpoint
            else:
                high = midpoint
            if high - low <= self.tolerance:
                break
        return float(low)

    def __repr__(self) -> str:
        return f"KLUCBPolicy(c={self.c}, tolerance={self.tolerance})"


class MOSSPolicy(UCBPolicy):
    """Minimax Optimal Strategy in the Stochastic case (MOSS)."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        horizon: int,
        c: float = 1.0,
        reward_scale: float = 1.0,
    ) -> None:
        validate_positive_integer(horizon, name="horizon")
        if horizon < n_arms:
            raise ValueError("horizon must be greater than or equal to n_arms")
        self.horizon = int(horizon)
        super().__init__(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
        )

    def _confidence_bonus(self) -> FloatArray:
        counts = np.maximum(self.counts, 1.0)
        log_terms = np.maximum(np.log(self.horizon / (self.n_arms * counts)), 0.0)
        return self.reward_scale * np.sqrt(self.c * log_terms / counts)

    def __repr__(self) -> str:
        return (
            f"MOSSPolicy(horizon={self.horizon}, c={self.c}, "
            f"reward_scale={self.reward_scale})"
        )


@dataclass(eq=False)
class UCBStats:
    """Diagnostic UCB state."""

    estimates: FloatArray
    counts: FloatArray
    bonuses: FloatArray


def _bernoulli_kl(p: float, q: float) -> float:
    epsilon = 1e-15
    clipped_p = float(np.clip(p, epsilon, 1.0 - epsilon))
    clipped_q = float(np.clip(q, epsilon, 1.0 - epsilon))
    return float(
        clipped_p * np.log(clipped_p / clipped_q)
        + (1.0 - clipped_p) * np.log((1.0 - clipped_p) / (1.0 - clipped_q))
    )


__all__ = ["KLUCBPolicy", "MOSSPolicy", "UCBPolicy", "UCBStats"]
