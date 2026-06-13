"""Bayesian UCB policies."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from pymab.policies.policy import ActionValuePolicy, FloatArray, Policy


class BernoulliBayesianUCBPolicy(ActionValuePolicy):
    """Bayesian UCB using Beta posterior quantiles."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        quantile: float = 0.95,
        **_: object,
    ) -> None:
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior and beta_prior must be positive")
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.quantile = float(quantile)
        self.successes: FloatArray
        self.failures: FloatArray
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        super().__init__(n_arms=int(arms), initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.successes = np.zeros(self.n_arms, dtype=float)
        self.failures = np.zeros(self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        # NumPy lacks a beta PPF. A large deterministic posterior sample is a
        # practical approximation for the quantile and keeps SciPy optional.
        samples = rng.beta(
            self.alpha_prior + self.successes[:, np.newaxis],
            self.beta_prior + self.failures[:, np.newaxis],
            size=(self.n_arms, 256),
        )
        upper_bounds = np.quantile(samples, self.quantile, axis=1)
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
        n_arms: int | None = None,
        n_bandits: int | None = None,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        reward_precision: float = 1.0,
        quantile: float = 0.95,
        **_: object,
    ) -> None:
        if prior_precision <= 0 or reward_precision <= 0:
            raise ValueError("precisions must be positive")
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.reward_precision = float(reward_precision)
        self.quantile = float(quantile)
        self.means: FloatArray
        self.precisions: FloatArray
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        super().__init__(n_arms=int(arms), initial_value=prior_mean)

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


def BayesianUCBPolicy(
    n_arms: int | None = None,
    n_bandits: int | None = None,
    *,
    reward_distribution: str = "gaussian",
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    prior_mean: float = 0.0,
    prior_precision: float = 1.0,
    reward_precision: float = 1.0,
    quantile: float = 0.95,
    **_: object,
) -> Policy:
    """Create a distribution-specific Bayesian UCB policy."""

    arms = n_arms if n_arms is not None else n_bandits
    if arms is None:
        raise TypeError("n_arms is required")
    if reward_distribution == "bernoulli":
        return BernoulliBayesianUCBPolicy(
            n_arms=int(arms),
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            quantile=quantile,
        )
    if reward_distribution in {"gaussian", "normal"}:
        return GaussianBayesianUCBPolicy(
            n_arms=int(arms),
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            reward_precision=reward_precision,
            quantile=quantile,
        )
    raise ValueError(f"{reward_distribution!r} cannot be used with Bayesian UCB")
