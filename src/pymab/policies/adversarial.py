"""Adversarial bandit policies."""

from __future__ import annotations

import numpy as np

from pymab.errors import ValidationError
from pymab.policies.policy import (
    ActionValuePolicy,
    validate_positive,
    validate_probability,
)
from pymab.types import FloatArray, PolicyCapabilities, RewardDomain


class EXP3Policy(ActionValuePolicy):
    """EXP3 for adversarial rewards in ``[0, 1]``.

    The policy maintains multiplicative weights over arms and updates the
    selected arm with an importance-weighted reward estimate. ``gamma`` mixes in
    uniform exploration and also serves as the default learning-rate scale.
    """

    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=frozenset({RewardDomain.BINARY, RewardDomain.UNIT_INTERVAL}),
    )

    def __init__(
        self,
        *,
        n_arms: int,
        gamma: float = 0.07,
        learning_rate: float | None = None,
    ) -> None:
        validate_probability(gamma, name="gamma")
        if gamma <= 0:
            raise ValueError("gamma must be in (0, 1]")
        if learning_rate is not None:
            validate_positive(learning_rate, name="learning_rate")
            if learning_rate > 1:
                raise ValueError("learning_rate must be in (0, 1]")
        self.gamma = float(gamma)
        self.learning_rate = float(gamma if learning_rate is None else learning_rate)
        self.log_weights: FloatArray
        self.last_probabilities: FloatArray
        super().__init__(n_arms=n_arms, initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.log_weights = np.zeros(self.n_arms, dtype=float)
        self.last_probabilities = np.full(self.n_arms, 1.0 / self.n_arms, dtype=float)

    def select_action(self, *, rng: np.random.Generator) -> int:
        probabilities = self.action_probabilities()
        self.last_probabilities = probabilities
        return int(rng.choice(self.n_arms, p=probabilities))

    def update(self, *, action: int, reward: float) -> None:
        validate_probability(float(reward), name="reward")
        self._validate_action(action)
        self.step += 1
        self.total_reward += float(reward)
        self.counts[action] += 1.0
        self._update_estimate(action=action, reward=float(reward))
        probability = float(self.last_probabilities[action])
        if probability <= 0 or not np.isfinite(probability):
            raise ValidationError(
                "selected-action probability must be positive and finite"
            )
        reward_hat = float(reward) / probability
        increment = (self.learning_rate * reward_hat) / self.n_arms
        if not np.isfinite(increment):
            raise ValidationError("EXP3 update exceeded floating-point range")
        self.log_weights[action] += increment
        self.log_weights -= np.max(self.log_weights)

    def action_probabilities(self) -> FloatArray:
        shifted = self.log_weights - np.max(self.log_weights)
        relative_weights = np.exp(shifted)
        total = float(np.sum(relative_weights))
        exploitation = (1.0 - self.gamma) * (relative_weights / total)
        exploration = self.gamma / self.n_arms
        probabilities = np.asarray(exploitation + exploration, dtype=float)
        probabilities /= np.sum(probabilities)
        return probabilities

    @property
    def weights(self) -> FloatArray:
        """Return relative multiplicative weights, normalized to a maximum of one."""

        return np.asarray(
            np.exp(self.log_weights - np.max(self.log_weights)), dtype=float
        )

    def recommend_action(self) -> int:
        return int(np.argmax(self.action_probabilities()))

    def __repr__(self) -> str:
        return f"EXP3Policy(gamma={self.gamma}, learning_rate={self.learning_rate})"


__all__ = ["EXP3Policy"]
