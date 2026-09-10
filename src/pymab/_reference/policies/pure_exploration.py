"""Pure-exploration policies for best-arm identification."""

from __future__ import annotations

import math

import numpy as np

from pymab.policies.policy import (
    ActionValuePolicy,
    validate_positive,
    validate_probability,
)
from pymab.types import (
    ALL_REWARD_DOMAINS,
    PolicyCapabilities,
    PolicyObjective,
    RewardDomain,
)


class SuccessiveEliminationPolicy(ActionValuePolicy):
    """Successive elimination for fixed-confidence best-arm identification."""

    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=ALL_REWARD_DOMAINS,
        objective=PolicyObjective.BEST_ARM,
    )

    def __init__(
        self,
        *,
        n_arms: int,
        delta: float = 0.05,
        confidence_scale: float = 1.0,
    ) -> None:
        validate_probability(delta, name="delta")
        if delta == 0:
            raise ValueError("delta must be positive")
        validate_positive(confidence_scale, name="confidence_scale")
        self.delta = float(delta)
        self.confidence_scale = float(confidence_scale)
        self.active: np.ndarray
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.active = np.ones(self.n_arms, dtype=bool)

    def select_action(self, *, rng: np.random.Generator) -> int:
        active_arms = np.flatnonzero(self.active)
        if active_arms.size == 1:
            return int(active_arms[0])
        active_counts = self.counts[active_arms]
        minimum_count = np.min(active_counts)
        candidates = active_arms[active_counts == minimum_count]
        return int(rng.choice(candidates))

    def update(self, *, action: int, reward: float) -> None:
        super().update(action=action, reward=float(reward))
        self._eliminate_confidently_suboptimal_arms()

    @property
    def best_arm(self) -> int:
        """Return the current best-arm recommendation."""

        active_arms = np.flatnonzero(self.active)
        if active_arms.size == 1:
            return int(active_arms[0])
        return int(active_arms[np.argmax(self.estimates[active_arms])])

    def recommend_action(self) -> int:
        return self.best_arm

    def _eliminate_confidently_suboptimal_arms(self) -> None:
        active_arms = np.flatnonzero(self.active)
        if active_arms.size <= 1 or np.any(self.counts[active_arms] == 0):
            return
        radii = self._confidence_radii()
        lower_bounds = self.estimates - radii
        upper_bounds = self.estimates + radii
        best_lower = float(np.max(lower_bounds[active_arms]))
        keep = upper_bounds[active_arms] >= best_lower
        if np.any(keep):
            self.active[active_arms] = keep

    def _confidence_radii(self) -> np.ndarray:
        counts = np.maximum(self.counts, 1.0)
        log_term = math.log((4.0 * self.n_arms * max(self.step, 1) ** 2) / self.delta)
        return self.confidence_scale * np.sqrt(log_term / (2.0 * counts))

    def __repr__(self) -> str:
        return (
            "SuccessiveEliminationPolicy("
            f"delta={self.delta}, confidence_scale={self.confidence_scale})"
        )


class MedianEliminationPolicy(ActionValuePolicy):
    """Median elimination for rewards bounded to the unit interval."""

    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=frozenset({RewardDomain.BINARY, RewardDomain.UNIT_INTERVAL}),
        objective=PolicyObjective.BEST_ARM,
    )

    def __init__(
        self,
        *,
        n_arms: int,
        epsilon: float = 0.1,
        delta: float = 0.05,
    ) -> None:
        validate_probability(epsilon, name="epsilon")
        validate_probability(delta, name="delta")
        if epsilon == 0:
            raise ValueError("epsilon must be positive")
        if delta == 0:
            raise ValueError("delta must be positive")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.phase_epsilon = float(epsilon / 4.0)
        self.phase_delta = float(delta / 2.0)
        self.active: np.ndarray
        self.phase_counts: np.ndarray
        self.phase_sums: np.ndarray
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.phase_epsilon = self.epsilon / 4.0
        self.phase_delta = self.delta / 2.0
        self.active = np.ones(self.n_arms, dtype=bool)
        self.phase_counts = np.zeros(self.n_arms, dtype=float)
        self.phase_sums = np.zeros(self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        active_arms = np.flatnonzero(self.active)
        if active_arms.size == 1:
            return int(active_arms[0])
        quota = self._phase_quota()
        under_sampled = active_arms[self.phase_counts[active_arms] < quota]
        candidates = under_sampled if under_sampled.size else active_arms
        phase_counts = self.phase_counts[candidates]
        minimum_count = np.min(phase_counts)
        tied = candidates[phase_counts == minimum_count]
        return int(rng.choice(tied))

    def update(self, *, action: int, reward: float) -> None:
        validate_probability(float(reward), name="reward")
        super().update(action=action, reward=float(reward))
        self.phase_counts[action] += 1.0
        self.phase_sums[action] += float(reward)
        self._complete_phase_if_ready()

    @property
    def best_arm(self) -> int:
        """Return the current best-arm recommendation."""

        active_arms = np.flatnonzero(self.active)
        if active_arms.size == 1:
            return int(active_arms[0])
        return int(active_arms[np.argmax(self.estimates[active_arms])])

    def recommend_action(self) -> int:
        return self.best_arm

    def _complete_phase_if_ready(self) -> None:
        active_arms = np.flatnonzero(self.active)
        if active_arms.size <= 1:
            return
        quota = self._phase_quota()
        if np.any(self.phase_counts[active_arms] < quota):
            return
        means = self.phase_sums[active_arms] / self.phase_counts[active_arms]
        median = float(np.median(means))
        keep = means >= median
        if np.any(keep):
            self.active[active_arms] = keep
        self.phase_counts[:] = 0.0
        self.phase_sums[:] = 0.0
        self.phase_epsilon *= 0.75
        self.phase_delta *= 0.5

    def _phase_quota(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    (4.0 / (self.phase_epsilon**2)) * math.log(3.0 / self.phase_delta)
                )
            ),
        )

    def __repr__(self) -> str:
        return f"MedianEliminationPolicy(epsilon={self.epsilon}, delta={self.delta})"


__all__ = ["MedianEliminationPolicy", "SuccessiveEliminationPolicy"]
