"""Linear contextual bandit policies."""

from __future__ import annotations

from typing import cast

import numpy as np

from pymab.policies.policy import (
    ContextualPolicy,
    FloatArray,
    choose_argmax,
    validate_probability,
)


class LinearEpsilonGreedyPolicy(ContextualPolicy):
    """Linear contextual policy with epsilon-greedy exploration."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        n_features: int | None = None,
        context_dim: int | None = None,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
        **_: object,
    ) -> None:
        validate_probability(epsilon, name="epsilon")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        arms = n_arms if n_arms is not None else n_bandits
        features = n_features if n_features is not None else context_dim
        if arms is None or features is None:
            raise TypeError("n_arms and n_features are required")
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.theta: FloatArray
        super().__init__(n_arms=int(arms), n_features=int(features))
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros((self.n_arms, self.n_features), dtype=float)

    def select_action(self, *, context: FloatArray, rng: np.random.Generator) -> int:
        self._validate_context(context)
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_arms))
        return choose_argmax(self.scores(context), rng)

    def update(self, *, action: int, reward: float, context: FloatArray) -> None:
        self._validate_action(action)
        self._validate_context(context)
        x = context[action]
        error = float(reward) - float(self.theta[action] @ x)
        self.theta[action] += self.learning_rate * error * x

    def scores(self, context: FloatArray) -> FloatArray:
        self._validate_context(context)
        return cast(
            FloatArray, np.einsum("ij,ij->i", context, self.theta).astype(float)
        )

    def __repr__(self) -> str:
        return (
            "LinearEpsilonGreedyPolicy("
            f"epsilon={self.epsilon}, learning_rate={self.learning_rate})"
        )


class LinUCBPolicy(ContextualPolicy):
    """Disjoint linear UCB for contextual bandits."""

    def __init__(
        self,
        *,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        l2: float = 1.0,
        **_: object,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.a: FloatArray
        self.b: FloatArray
        super().__init__(n_arms=n_arms, n_features=n_features)
        self.reset()

    def reset(self) -> None:
        eye = np.eye(self.n_features, dtype=float)
        self.a = np.repeat((self.l2 * eye)[np.newaxis, :, :], self.n_arms, axis=0)
        self.b = np.zeros((self.n_arms, self.n_features), dtype=float)

    def select_action(self, *, context: FloatArray, rng: np.random.Generator) -> int:
        self._validate_context(context)
        return choose_argmax(self.upper_confidence_bounds(context), rng)

    def update(self, *, action: int, reward: float, context: FloatArray) -> None:
        self._validate_action(action)
        self._validate_context(context)
        x = context[action]
        self.a[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def upper_confidence_bounds(self, context: FloatArray) -> FloatArray:
        self._validate_context(context)
        values = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            a_inv = np.linalg.inv(self.a[arm])
            theta = a_inv @ self.b[arm]
            x = context[arm]
            uncertainty = np.sqrt(float(x @ a_inv @ x))
            values[arm] = float(theta @ x) + self.alpha * uncertainty
        return values

    def __repr__(self) -> str:
        return f"LinUCBPolicy(alpha={self.alpha}, l2={self.l2})"


class LinearThompsonSamplingPolicy(ContextualPolicy):
    """Bayesian linear Thompson Sampling for contextual bandits."""

    def __init__(
        self,
        *,
        n_arms: int,
        n_features: int,
        exploration_scale: float = 1.0,
        l2: float = 1.0,
        **_: object,
    ) -> None:
        if exploration_scale <= 0:
            raise ValueError("exploration_scale must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")
        self.exploration_scale = float(exploration_scale)
        self.l2 = float(l2)
        self.a: FloatArray
        self.b: FloatArray
        super().__init__(n_arms=n_arms, n_features=n_features)
        self.reset()

    def reset(self) -> None:
        eye = np.eye(self.n_features, dtype=float)
        self.a = np.repeat((self.l2 * eye)[np.newaxis, :, :], self.n_arms, axis=0)
        self.b = np.zeros((self.n_arms, self.n_features), dtype=float)

    def select_action(self, *, context: FloatArray, rng: np.random.Generator) -> int:
        self._validate_context(context)
        samples = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            a_inv = np.linalg.inv(self.a[arm])
            mean = a_inv @ self.b[arm]
            cov = (self.exploration_scale**2) * a_inv
            theta_sample = rng.multivariate_normal(mean, cov)
            samples[arm] = float(theta_sample @ context[arm])
        return int(np.argmax(samples))

    def update(self, *, action: int, reward: float, context: FloatArray) -> None:
        self._validate_action(action)
        self._validate_context(context)
        x = context[action]
        self.a[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def __repr__(self) -> str:
        return (
            "LinearThompsonSamplingPolicy("
            f"exploration_scale={self.exploration_scale}, l2={self.l2})"
        )


class LogisticContextualBanditPolicy(ContextualPolicy):
    """Online logistic contextual bandit for Bernoulli rewards.

    Each arm has an independent logistic model. The policy predicts click or
    conversion probability for each arm, explores with epsilon-greedy sampling,
    and updates the selected arm using one stochastic-gradient step.
    """

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        n_features: int | None = None,
        context_dim: int | None = None,
        epsilon: float = 0.05,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        **_: object,
    ) -> None:
        validate_probability(epsilon, name="epsilon")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        arms = n_arms if n_arms is not None else n_bandits
        features = n_features if n_features is not None else context_dim
        if arms is None or features is None:
            raise TypeError("n_arms and n_features are required")
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.theta: FloatArray
        super().__init__(n_arms=int(arms), n_features=int(features))
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros((self.n_arms, self.n_features), dtype=float)

    def select_action(self, *, context: FloatArray, rng: np.random.Generator) -> int:
        self._validate_context(context)
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_arms))
        return choose_argmax(self.predicted_probabilities(context), rng)

    def update(self, *, action: int, reward: float, context: FloatArray) -> None:
        self._validate_action(action)
        self._validate_context(context)
        validate_probability(float(reward), name="reward")
        x = context[action]
        probability = float(_sigmoid(float(self.theta[action] @ x)))
        gradient = (float(reward) - probability) * x - self.l2 * self.theta[action]
        self.theta[action] += self.learning_rate * gradient

    def predicted_probabilities(self, context: FloatArray) -> FloatArray:
        self._validate_context(context)
        logits = np.einsum("ij,ij->i", context, self.theta).astype(float)
        return cast(FloatArray, _sigmoid(logits))

    def __repr__(self) -> str:
        return (
            "LogisticContextualBanditPolicy("
            f"epsilon={self.epsilon}, learning_rate={self.learning_rate}, "
            f"l2={self.l2})"
        )


ContextualBanditPolicy = LinearEpsilonGreedyPolicy


def _sigmoid(values: FloatArray | float) -> FloatArray | float:
    clipped = np.clip(values, -35.0, 35.0)
    return cast(FloatArray | float, 1.0 / (1.0 + np.exp(-clipped)))
