"""Change-detection policies for abruptly non-stationary rewards."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymab.policies.policy import validate_positive, validate_positive_integer
from pymab.policies.ucb import UCBPolicy
from pymab.types import FloatArray


class ChangePointUCBPolicy(UCBPolicy):
    """UCB with per-arm change detection and local arm resets."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        c: float = 2.0,
        reward_scale: float = 1.0,
        detector: str = "cusum",
        threshold: float = 5.0,
        drift: float = 0.05,
        min_observations: int = 20,
    ) -> None:
        if detector not in {"cusum", "page_hinkley"}:
            raise ValueError("detector must be 'cusum' or 'page_hinkley'")
        validate_positive(threshold, name="threshold")
        validate_positive(drift, name="drift", allow_zero=True)
        validate_positive_integer(min_observations, name="min_observations")
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
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
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


__all__ = ["CUSUMUCBPolicy", "ChangePointUCBPolicy", "PageHinkleyUCBPolicy"]
