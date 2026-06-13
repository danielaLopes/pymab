"""Upper-confidence-bound policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, cast

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


class KLUCBPolicy(UCBPolicy):
    """KL-UCB for Bernoulli rewards.

    Rewards must be binary. The index for each arm is the largest Bernoulli
    mean ``q`` whose KL divergence from the empirical mean is within the
    standard logarithmic confidence budget.
    """

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        c: float = 3.0,
        tolerance: float = 1e-6,
        max_iterations: int = 32,
        **kwargs: object,
    ) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            initial_value=initial_value,
            optimistic_initialization=optimistic_initialization,
            c=c,
            **kwargs,
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
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        horizon: int,
        c: float = 1.0,
        **kwargs: object,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            initial_value=initial_value,
            optimistic_initialization=optimistic_initialization,
            c=c,
            **kwargs,
        )

    def _confidence_bonus(self) -> FloatArray:
        counts = np.maximum(self.counts, 1.0)
        log_terms = np.maximum(np.log(self.horizon / (self.n_arms * counts)), 0.0)
        return cast(FloatArray, np.sqrt(self.c * log_terms / counts))

    def __repr__(self) -> str:
        return f"MOSSPolicy(horizon={self.horizon}, c={self.c})"


MOSSUCBPolicy = MOSSPolicy


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


class ChangePointUCBPolicy(UCBPolicy):
    """UCB with per-arm change detection and local arm resets.

    ``detector="cusum"`` is sensitive to abrupt mean shifts through two-sided
    cumulative sums. ``detector="page_hinkley"`` uses a Page-Hinkley statistic
    to reset an arm after sustained positive drift. Both detectors reset only
    the arm whose stream changed, preserving information about other arms.
    """

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        c: float = 2.0,
        detector: str = "cusum",
        threshold: float = 5.0,
        drift: float = 0.05,
        min_observations: int = 20,
        **kwargs: object,
    ) -> None:
        if detector not in {"cusum", "page_hinkley"}:
            raise ValueError("detector must be 'cusum' or 'page_hinkley'")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if drift < 0:
            raise ValueError("drift must be non-negative")
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        self.detector = detector
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.min_observations = int(min_observations)
        self.detector_counts: FloatArray
        self.detector_means: FloatArray
        self.positive_cusum: FloatArray
        self.negative_cusum: FloatArray
        self.ph_cumulative: FloatArray
        self.ph_minimum: FloatArray
        self.change_counts: FloatArray
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
        self.detector_counts = np.zeros(self.n_arms, dtype=float)
        self.detector_means = np.zeros(self.n_arms, dtype=float)
        self.positive_cusum = np.zeros(self.n_arms, dtype=float)
        self.negative_cusum = np.zeros(self.n_arms, dtype=float)
        self.ph_cumulative = np.zeros(self.n_arms, dtype=float)
        self.ph_minimum = np.zeros(self.n_arms, dtype=float)
        self.change_counts = np.zeros(self.n_arms, dtype=float)

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        value = float(reward)
        previous_mean = self.detector_means[action]
        super().update(action=action, reward=value)
        changed = self._update_detector(
            action=action,
            reward=value,
            previous_mean=float(previous_mean),
        )
        if changed:
            self._reset_arm(action=action, reward=value)

    def _update_detector(
        self, *, action: int, reward: float, previous_mean: float
    ) -> bool:
        self.detector_counts[action] += 1.0
        count = self.detector_counts[action]
        self.detector_means[action] += (reward - self.detector_means[action]) / count
        if count < self.min_observations:
            return False

        residual = reward - previous_mean
        if self.detector == "cusum":
            self.positive_cusum[action] = max(
                0.0, self.positive_cusum[action] + residual - self.drift
            )
            self.negative_cusum[action] = max(
                0.0, self.negative_cusum[action] - residual - self.drift
            )
            return bool(
                self.positive_cusum[action] > self.threshold
                or self.negative_cusum[action] > self.threshold
            )

        centered = residual - self.drift
        self.ph_cumulative[action] += centered
        self.ph_minimum[action] = min(
            self.ph_minimum[action], self.ph_cumulative[action]
        )
        return bool(
            self.ph_cumulative[action] - self.ph_minimum[action] > self.threshold
        )

    def _reset_arm(self, *, action: int, reward: float) -> None:
        self.change_counts[action] += 1.0
        self.counts[action] = 1.0
        self.estimates[action] = reward
        self.detector_counts[action] = 1.0
        self.detector_means[action] = reward
        self.positive_cusum[action] = 0.0
        self.negative_cusum[action] = 0.0
        self.ph_cumulative[action] = 0.0
        self.ph_minimum[action] = 0.0

    def __repr__(self) -> str:
        return (
            "ChangePointUCBPolicy("
            f"detector={self.detector!r}, threshold={self.threshold}, "
            f"drift={self.drift})"
        )


class CUSUMUCBPolicy(ChangePointUCBPolicy):
    """CUSUM-triggered resetting UCB."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(detector="cusum", **kwargs)


class PageHinkleyUCBPolicy(ChangePointUCBPolicy):
    """Page-Hinkley-triggered resetting UCB."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(detector="page_hinkley", **kwargs)


@dataclass
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
