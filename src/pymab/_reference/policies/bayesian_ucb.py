"""Bayesian UCB policies."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from pymab.policies.policy import ActionValuePolicy, validate_positive
from pymab.types import FloatArray, PolicyCapabilities, RewardDomain


class BernoulliBayesianUCBPolicy(ActionValuePolicy):
    """Bayesian UCB using Beta posterior quantiles."""

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
        quantile: float = 0.95,
    ) -> None:
        validate_positive(alpha_prior, name="alpha_prior")
        validate_positive(beta_prior, name="beta_prior")
        if not np.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.quantile = float(quantile)
        self.successes: FloatArray
        self.failures: FloatArray
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.successes = np.zeros(self.n_arms, dtype=float)
        self.failures = np.zeros(self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        try:
            from scipy.stats import beta as beta_distribution
        except ImportError as exc:
            raise ImportError(
                "Install pymab[bayes] to use BernoulliBayesianUCBPolicy"
            ) from exc
        upper_bounds = beta_distribution.ppf(
            self.quantile,
            self.alpha_prior + self.successes,
            self.beta_prior + self.failures,
        )
        return int(np.argmax(upper_bounds))

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Bernoulli Bayesian UCB requires binary rewards")
        super().update(action=action, reward=float(reward))
        if reward > 0:
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0

    def __repr__(self) -> str:
        return (
            "BernoulliBayesianUCBPolicy("
            f"quantile={self.quantile}, alpha_prior={self.alpha_prior}, "
            f"beta_prior={self.beta_prior})"
        )


class GaussianBayesianUCBPolicy(ActionValuePolicy):
    """Bayesian UCB for Gaussian rewards with known observation variance."""

    def __init__(
        self,
        *,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        reward_precision: float = 1.0,
        quantile: float = 0.95,
    ) -> None:
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        validate_positive(prior_precision, name="prior_precision")
        validate_positive(reward_precision, name="reward_precision")
        if not np.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.reward_precision = float(reward_precision)
        self.quantile = float(quantile)
        self.means: FloatArray
        self.precisions: FloatArray
        super().__init__(n_arms=n_arms, initial_value=prior_mean)

    def reset(self) -> None:
        super().reset()
        self.means = np.full(self.n_arms, self.prior_mean, dtype=float)
        self.precisions = np.full(self.n_arms, self.prior_precision, dtype=float)
        self.estimates = self.means.copy()

    def select_action(self, *, rng: np.random.Generator) -> int:
        z = NormalDist().inv_cdf(self.quantile)
        upper_bounds = self.means + z / np.sqrt(self.precisions)
        return int(np.argmax(upper_bounds))

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
            "GaussianBayesianUCBPolicy("
            f"quantile={self.quantile}, prior_mean={self.prior_mean}, "
            f"reward_precision={self.reward_precision})"
        )


__all__ = ["BernoulliBayesianUCBPolicy", "GaussianBayesianUCBPolicy"]
