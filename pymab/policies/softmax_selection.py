"""Softmax action-selection policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, softmax


class SoftmaxPolicy(ActionValuePolicy):
    """Sample actions according to a softmax over estimated values."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        temperature: float = 1.0,
        **_: object,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        init = optimistic_initialization if initial_value is None else initial_value
        super().__init__(n_arms=int(arms), initial_value=float(init))
        self.temperature = float(temperature)

    def action_probabilities(self) -> np.ndarray:
        """Return the current softmax action probabilities."""

        return softmax(self.estimates, self.temperature)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_arms, p=self.action_probabilities()))

    def __repr__(self) -> str:
        return f"SoftmaxPolicy(temperature={self.temperature})"


SoftmaxSelectionPolicy = SoftmaxPolicy
