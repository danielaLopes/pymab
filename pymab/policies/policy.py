"""Policy interfaces and reusable policy state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Policy(ABC):
    """Base class for non-contextual bandit policies."""

    n_arms: int

    def __post_init__(self) -> None:
        if self.n_arms <= 0:
            raise ValueError("n_arms must be positive")

    @abstractmethod
    def select_action(self, *, rng: np.random.Generator) -> int:
        """Choose an action index."""

    @abstractmethod
    def update(self, *, action: int, reward: float) -> None:
        """Update policy state after observing a reward."""

    @abstractmethod
    def reset(self) -> None:
        """Reset learned state."""

    def clone(self) -> Policy:
        """Return an independent copy suitable for a new episode."""

        return deepcopy(self)

    @property
    def n_bandits(self) -> int:
        """Backward-compatible alias for ``n_arms``."""

        return self.n_arms

    def _validate_action(self, action: int) -> None:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")


@dataclass
class ContextualPolicy(ABC):
    """Base class for linear contextual bandit policies."""

    n_arms: int
    n_features: int

    def __post_init__(self) -> None:
        if self.n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")

    @abstractmethod
    def select_action(self, *, context: FloatArray, rng: np.random.Generator) -> int:
        """Choose an action index for a context matrix."""

    @abstractmethod
    def update(self, *, action: int, reward: float, context: FloatArray) -> None:
        """Update policy state after a contextual reward."""

    @abstractmethod
    def reset(self) -> None:
        """Reset learned state."""

    def clone(self) -> ContextualPolicy:
        """Return an independent copy suitable for a new episode."""

        return deepcopy(self)

    @property
    def n_bandits(self) -> int:
        """Backward-compatible alias for ``n_arms``."""

        return self.n_arms

    def _validate_context(self, context: FloatArray) -> None:
        if context.shape != (self.n_arms, self.n_features):
            raise ValueError("context must have shape (n_arms, n_features)")

    def _validate_action(self, action: int) -> None:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")


@dataclass
class ActionValuePolicy(Policy):
    """Base class for policies that estimate one value per action."""

    initial_value: float = 0.0
    step: int = field(init=False, default=0)
    total_reward: float = field(init=False, default=0.0)
    counts: FloatArray = field(init=False)
    estimates: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset()

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        self.step += 1
        self.total_reward += float(reward)
        self.counts[action] += 1.0
        self._update_estimate(action=action, reward=float(reward))

    def reset(self) -> None:
        self.step = 0
        self.total_reward = 0.0
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.estimates = np.full(self.n_arms, self.initial_value, dtype=float)

    def _update_estimate(self, *, action: int, reward: float) -> None:
        self.estimates[action] += (reward - self.estimates[action]) / self.counts[
            action
        ]

    @property
    def times_selected(self) -> FloatArray:
        """Backward-compatible alias for action counts."""

        return self.counts

    @property
    def actions_estimated_reward(self) -> FloatArray:
        """Backward-compatible alias for action-value estimates."""

        return self.estimates

    @property
    def current_step(self) -> int:
        """Backward-compatible alias for ``step``."""

        return self.step


def softmax(values: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Numerically stable softmax."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    return cast(FloatArray, exp_values / np.sum(exp_values))


def validate_probability(value: float, *, name: str) -> None:
    """Validate a scalar probability."""

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def choose_argmax(values: FloatArray, rng: np.random.Generator) -> int:
    """Break argmax ties uniformly at random."""

    candidates = np.flatnonzero(values == np.max(values))
    return int(rng.choice(candidates))


def no_context_func() -> None:
    """Compatibility placeholder for the old no-context callback."""

    return None
