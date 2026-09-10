"""Greedy bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, choose_argmax


class GreedyPolicy(ActionValuePolicy):
    """Always select the action with the highest estimated value."""

    def __init__(
        self,
        *,
        n_arms: int,
        initial_value: float = 0.0,
    ) -> None:
        super().__init__(n_arms=n_arms, initial_value=float(initial_value))

    def select_action(self, *, rng: np.random.Generator) -> int:
        return choose_argmax(self.estimates, rng)

    def __repr__(self) -> str:
        return f"GreedyPolicy(initial_value={self.initial_value})"


__all__ = ["GreedyPolicy"]
