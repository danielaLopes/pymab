"""Stationary Thompson sampling policies."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, validate_positive
from pymab.types import FloatArray, PolicyCapabilities, RewardDomain


class BernoulliThompsonSamplingPolicy(ActionValuePolicy):
    """Thompson sampling with Beta-Bernoulli posteriors."""

    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=frozenset({RewardDomain.BINARY}),
    )

    def __init__(
        self,
        *,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        validate_positive(alpha_prior, name="alpha_prior")
        validate_positive(beta_prior, name="beta_prior")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.successes: FloatArray
        self.failures: FloatArray
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.successes = np.zeros(self.n_arms, dtype=float)
        self.failures = np.zeros(self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        samples = rng.beta(
            self.alpha_prior + self.successes,
            self.beta_prior + self.failures,
        )
        return int(np.argmax(samples))

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Bernoulli Thompson Sampling requires binary rewards")
        super().update(action=action, reward=float(reward))
        if reward > 0:
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0

    def __repr__(self) -> str:
        return (
            "BernoulliThompsonSamplingPolicy("
            f"alpha_prior={self.alpha_prior}, beta_prior={self.beta_prior})"
        )


class GaussianThompsonSamplingPolicy(ActionValuePolicy):
    """Thompson sampling for Gaussian rewards with known observation variance."""

    def __init__(
        self,
        *,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        reward_precision: float = 1.0,
    ) -> None:
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        validate_positive(prior_precision, name="prior_precision")
        validate_positive(reward_precision, name="reward_precision")
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.reward_precision = float(reward_precision)
        self.means: FloatArray
        self.precisions: FloatArray
        super().__init__(n_arms=n_arms, initial_value=prior_mean)

    def reset(self) -> None:
        super().reset()
        self.means = np.full(self.n_arms, self.prior_mean, dtype=float)
        self.precisions = np.full(self.n_arms, self.prior_precision, dtype=float)
        self.estimates = self.means.copy()

    def select_action(self, *, rng: np.random.Generator) -> int:
        samples = rng.normal(self.means, 1.0 / np.sqrt(self.precisions))
        return int(np.argmax(samples))

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.step += 1
        self.total_reward += float(reward)
        self.counts[action] += 1.0
        precision = self.precisions[action] + self.reward_precision
        mean = (
            self.precisions[action] * self.means[action]
            + self.reward_precision * float(reward)
        ) / precision
        self.precisions[action] = precision
        self.means[action] = mean
        self.estimates[action] = mean

    def __repr__(self) -> str:
        return (
            "GaussianThompsonSamplingPolicy("
            f"prior_mean={self.prior_mean}, reward_precision={self.reward_precision})"
        )


__all__ = ["BernoulliThompsonSamplingPolicy", "GaussianThompsonSamplingPolicy"]
