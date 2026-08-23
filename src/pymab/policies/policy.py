"""Policy interfaces and reusable policy state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import cast

import numpy as np

from pymab.types import (
    ALL_REWARD_DOMAINS,
    FloatArray,
    PolicyCapabilities,
)


@dataclass(eq=False)
class Policy(ABC):
    """Base class for non-contextual bandit policies."""

    n_arms: int
    capabilities = PolicyCapabilities(
        contextual=False,
        reward_domains=ALL_REWARD_DOMAINS,
    )

    def __post_init__(self) -> None:
        validate_positive_integer(self.n_arms, name="n_arms")

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
        """Create a fresh policy with the same configuration.

        The default deep-copies configuration and then resets learned state.
        Policies that own non-copyable resources must override this method.
        """

        result = deepcopy(self)
        result.reset()
        return result

    def recommend_action(self) -> int:
        """Return the current best-arm recommendation without exploration."""

        raise NotImplementedError(
            f"{type(self).__name__} does not expose a best-arm recommendation"
        )

    def _validate_action(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= int(action) < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")


@dataclass(eq=False)
class ContextualPolicy(ABC):
    """Base class for linear contextual bandit policies."""

    n_arms: int
    n_features: int
    capabilities = PolicyCapabilities(
        contextual=True,
        reward_domains=ALL_REWARD_DOMAINS,
    )

    def __post_init__(self) -> None:
        validate_positive_integer(self.n_arms, name="n_arms")
        validate_positive_integer(self.n_features, name="n_features")

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
        """Create a fresh policy with the same immutable configuration."""

        result = deepcopy(self)
        result.reset()
        return result

    def recommend_action(self, *, context: FloatArray) -> int:
        """Return the current recommendation without exploratory sampling."""

        raise NotImplementedError(
            f"{type(self).__name__} does not expose a recommendation"
        )

    def _validate_context(self, context: FloatArray) -> None:
        if context.shape != (self.n_arms, self.n_features):
            raise ValueError("context must have shape (n_arms, n_features)")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")

    def _validate_action(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= int(action) < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")


@dataclass(eq=False)
class ActionValuePolicy(Policy):
    """Base class for policies that estimate one value per action."""

    initial_value: float = 0.0
    step: int = field(init=False, default=0)
    total_reward: float = field(init=False, default=0.0)
    counts: FloatArray = field(init=False)
    estimates: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.initial_value):
            raise ValueError("initial_value must be finite")
        self.reset()

    def update(self, *, action: int, reward: float) -> None:
        self._validate_action(action)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
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

    def recommend_action(self) -> int:
        return int(np.argmax(self.estimates))


def softmax(values: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Numerically stable softmax."""

    validate_positive(temperature, name="temperature")
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite 1D array")
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    return cast(FloatArray, exp_values / np.sum(exp_values))


def validate_probability(value: float, *, name: str) -> None:
    """Validate a scalar probability."""

    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_positive(value: float, *, name: str, allow_zero: bool = False) -> None:
    """Validate a finite positive or non-negative scalar."""

    invalid_sign = value < 0 if allow_zero else value <= 0
    if not np.isfinite(value) or invalid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def validate_positive_integer(value: int, *, name: str) -> None:
    """Validate an integer count used to size arrays or loops."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def choose_argmax(values: FloatArray, rng: np.random.Generator) -> int:
    """Break argmax ties uniformly at random."""

    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite 1D array")
    candidates = np.flatnonzero(values == np.max(values))
    return int(rng.choice(candidates))


__all__ = [
    "ActionValuePolicy",
    "ContextualPolicy",
    "Policy",
    "choose_argmax",
    "softmax",
    "validate_positive",
    "validate_positive_integer",
    "validate_probability",
]
