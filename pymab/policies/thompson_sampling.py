"""Thompson sampling policies."""

from __future__ import annotations

from collections import deque

import numpy as np

from pymab.policies.policy import ActionValuePolicy, FloatArray, Policy


class BernoulliThompsonSamplingPolicy(ActionValuePolicy):
    """Thompson sampling with Beta-Bernoulli posteriors."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        **_: object,
    ) -> None:
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior and beta_prior must be positive")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
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
        n_arms: int | None = None,
        n_bandits: int | None = None,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        reward_precision: float = 1.0,
        **_: object,
    ) -> None:
        if prior_precision <= 0 or reward_precision <= 0:
            raise ValueError("precisions must be positive")
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.reward_precision = float(reward_precision)
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
        samples = rng.normal(self.means, 1.0 / np.sqrt(self.precisions))
        return int(np.argmax(samples))

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
            "GaussianThompsonSamplingPolicy("
            f"prior_mean={self.prior_mean}, reward_precision={self.reward_precision})"
        )


class SlidingWindowBernoulliThompsonSamplingPolicy(BernoulliThompsonSamplingPolicy):
    """Beta-Bernoulli Thompson Sampling over a fixed recent reward window."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
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
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self._windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    def update(self, *, action: int, reward: float) -> None:
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Sliding-window Thompson Sampling requires binary rewards")
        self._validate_action(action)
        self.step += 1
        self.total_reward += float(reward)
        self._windows[action].append(float(reward))
        self._refresh_posterior_from_windows()

    def _refresh_posterior_from_windows(self) -> None:
        for arm, rewards in enumerate(self._windows):
            values = np.array(rewards, dtype=float)
            self.counts[arm] = float(values.size)
            self.successes[arm] = float(np.sum(values)) if values.size else 0.0
            self.failures[arm] = float(values.size - self.successes[arm])
            self.estimates[arm] = (
                float(np.mean(values)) if values.size else self.initial_value
            )

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
        n_arms: int | None = None,
        n_bandits: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        discount_factor: float = 0.95,
        **kwargs: object,
    ) -> None:
        if not 0 < discount_factor < 1:
            raise ValueError("discount_factor must be in (0, 1)")
        self.discount_factor = float(discount_factor)
        super().__init__(
            n_arms=n_arms,
            n_bandits=n_bandits,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            **kwargs,
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


def ThompsonSamplingPolicy(
    n_arms: int | None = None,
    n_bandits: int | None = None,
    *,
    reward_distribution: str = "gaussian",
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    prior_mean: float = 0.0,
    prior_precision: float = 1.0,
    reward_precision: float = 1.0,
    **_: object,
) -> Policy:
    """Create a distribution-specific Thompson Sampling policy."""

    arms = n_arms if n_arms is not None else n_bandits
    if arms is None:
        raise TypeError("n_arms is required")
    if reward_distribution == "bernoulli":
        return BernoulliThompsonSamplingPolicy(
            n_arms=int(arms),
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )
    if reward_distribution in {"gaussian", "normal"}:
        return GaussianThompsonSamplingPolicy(
            n_arms=int(arms),
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            reward_precision=reward_precision,
        )
    raise ValueError(f"{reward_distribution!r} cannot be used with Thompson Sampling")
