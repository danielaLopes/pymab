"""Softmax action-selection policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, softmax, validate_positive


class SoftmaxPolicy(ActionValuePolicy):
    """Sample actions according to a softmax over estimated values."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        validate_positive(temperature, name="temperature")
        super().__init__(n_arms=n_arms, initial_value=float(initial_value))
        self.temperature = float(temperature)

    def action_probabilities(self) -> np.ndarray:
        """Return the current softmax action probabilities."""

        return softmax(self.estimates, self.temperature)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_arms, p=self.action_probabilities()))

    def __repr__(self) -> str:
        return f"SoftmaxPolicy(temperature={self.temperature})"


__all__ = ["SoftmaxPolicy"]
