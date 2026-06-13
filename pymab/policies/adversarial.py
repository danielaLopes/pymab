"""Adversarial bandit policies."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, FloatArray, validate_probability


class EXP3Policy(ActionValuePolicy):
    """EXP3 for adversarial rewards in ``[0, 1]``.

    The policy maintains multiplicative weights over arms and updates the
    selected arm with an importance-weighted reward estimate. ``gamma`` mixes in
    uniform exploration and also serves as the default learning-rate scale.
    """

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        gamma: float = 0.07,
        learning_rate: float | None = None,
        **_: object,
    ) -> None:
        validate_probability(gamma, name="gamma")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        self.gamma = float(gamma)
        self.learning_rate = float(gamma if learning_rate is None else learning_rate)
        self.weights: FloatArray
        self.last_probabilities: FloatArray
        super().__init__(n_arms=int(arms), initial_value=0.0)

    def reset(self) -> None:
        super().reset()
        self.weights = np.ones(self.n_arms, dtype=float)
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
        probability = max(float(self.last_probabilities[action]), 1e-12)
        reward_hat = float(reward) / probability
        growth = np.exp((self.learning_rate * reward_hat) / self.n_arms)
        self.weights[action] *= growth
        self.weights /= np.max(self.weights)

    def action_probabilities(self) -> FloatArray:
        total = float(np.sum(self.weights))
        if total <= 0 or not np.isfinite(total):
            self.weights = np.ones(self.n_arms, dtype=float)
            total = float(self.n_arms)
        exploitation = (1.0 - self.gamma) * (self.weights / total)
        exploration = self.gamma / self.n_arms
        return np.asarray(exploitation + exploration, dtype=float)

    def __repr__(self) -> str:
        return f"EXP3Policy(gamma={self.gamma}, learning_rate={self.learning_rate})"
