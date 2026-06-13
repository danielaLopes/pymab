"""Greedy bandit policy."""

from __future__ import annotations

import numpy as np

from pymab.policies.policy import ActionValuePolicy, choose_argmax


class GreedyPolicy(ActionValuePolicy):
    """Always select the action with the highest estimated value."""

    def __init__(
        self,
        *,
        n_arms: int | None = None,
        n_bandits: int | None = None,
        initial_value: float | None = None,
        optimistic_initialization: float = 0.0,
        **_: object,
    ) -> None:
        arms = n_arms if n_arms is not None else n_bandits
        if arms is None:
            raise TypeError("n_arms is required")
        init = optimistic_initialization if initial_value is None else initial_value
        super().__init__(n_arms=int(arms), initial_value=float(init))

    def select_action(self, *, rng: np.random.Generator) -> int:
        return choose_argmax(self.estimates, rng)

    def __repr__(self) -> str:
        return f"GreedyPolicy(initial_value={self.initial_value})"
