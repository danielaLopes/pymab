"""Bandit environments and non-stationary dynamics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import numpy as np

from pymab.distributions import FloatArray, RewardDistribution, resolve_distribution


class EnvironmentChangeType(Enum):
    """Built-in non-stationary environment dynamics."""

    STATIONARY = "stationary"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    RANDOM_ARM_SWAPPING = "random_arm_swapping"


class EnvironmentDynamics(ABC):
    """Strategy for mutating true arm values over time."""

    @abstractmethod
    def apply(
        self, q_values: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        """Return the next true arm values."""


@dataclass(frozen=True)
class StationaryDynamics(EnvironmentDynamics):
    """Keep true rewards fixed."""

    def apply(
        self, q_values: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        return q_values


@dataclass(frozen=True)
class GradualDrift(EnvironmentDynamics):
    """Add Gaussian noise to each arm on every step."""

    change_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.change_rate < 0:
            raise ValueError("change_rate must be non-negative")

    def apply(
        self, q_values: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        return q_values + rng.normal(0.0, self.change_rate, size=q_values.shape)


@dataclass(frozen=True)
class AbruptShift(EnvironmentDynamics):
    """Shift all arm values periodically."""

    change_frequency: int = 100
    change_magnitude: float = 0.5
    shift_at_step_zero: bool = False

    def __post_init__(self) -> None:
        if self.change_frequency <= 0:
            raise ValueError("change_frequency must be positive")
        if self.change_magnitude < 0:
            raise ValueError("change_magnitude must be non-negative")

    def apply(
        self, q_values: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        if step == 0 and not self.shift_at_step_zero:
            return q_values
        if step % self.change_frequency == 0:
            return q_values + rng.normal(
                0.0, self.change_magnitude, size=q_values.shape
            )
        return q_values


@dataclass(frozen=True)
class RandomArmSwap(EnvironmentDynamics):
    """Randomly permute arm values with a fixed step probability."""

    shift_probability: float = 0.2

    def __post_init__(self) -> None:
        if not 0 <= self.shift_probability <= 1:
            raise ValueError("shift_probability must be in [0, 1]")

    def apply(
        self, q_values: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        if rng.random() < self.shift_probability:
            return rng.permutation(q_values).astype(float)
        return q_values


@dataclass
class BanditEnvironment:
    """K-armed bandit environment.

    The environment owns the true action values and samples rewards. Policies
    only observe action indices, rewards, and optional contexts.
    """

    q_values: FloatArray
    reward_distribution: RewardDistribution = field(
        default_factory=lambda: resolve_distribution("gaussian")
    )
    dynamics: EnvironmentDynamics = field(default_factory=StationaryDynamics)

    def __post_init__(self) -> None:
        values = np.asarray(self.q_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("q_values must be a non-empty 1D array")
        self.q_values = values

    @classmethod
    def from_distribution(
        cls,
        *,
        n_arms: int,
        reward_distribution: str | RewardDistribution = "gaussian",
        q_mean: float = 0.0,
        q_scale: float = 1.0,
        reward_scale: float = 1.0,
        dynamics: EnvironmentDynamics | None = None,
        rng: np.random.Generator | None = None,
    ) -> BanditEnvironment:
        """Create an environment with generated true values."""

        generator = np.random.default_rng() if rng is None else rng
        distribution = resolve_distribution(
            reward_distribution, reward_scale=reward_scale
        )
        q_values = distribution.initial_values(
            mean=q_mean, scale=q_scale, n_arms=n_arms, rng=generator
        )
        return cls(
            q_values=q_values,
            reward_distribution=distribution,
            dynamics=StationaryDynamics() if dynamics is None else dynamics,
        )

    @property
    def n_arms(self) -> int:
        """Number of available actions."""

        return int(self.q_values.size)

    @property
    def optimal_action(self) -> int:
        """Index of the currently best action."""

        return int(np.argmax(self.q_values))

    @property
    def optimal_value(self) -> float:
        """Expected reward of the currently best action."""

        return float(np.max(self.q_values))

    def expected_reward(self, action: int) -> float:
        """Return the true expected reward for an action."""

        self._validate_action(action)
        return float(self.q_values[action])

    def step(self, action: int, *, rng: np.random.Generator) -> float:
        """Sample a reward for an action."""

        self._validate_action(action)
        return self.reward_distribution.sample_one(float(self.q_values[action]), rng)

    def advance(self, *, step: int, rng: np.random.Generator) -> None:
        """Apply non-stationary dynamics for a simulation step."""

        self.q_values = np.asarray(
            self.dynamics.apply(self.q_values, step=step, rng=rng), dtype=float
        )

    def copy(self) -> BanditEnvironment:
        """Create an independent environment with the same current state."""

        return BanditEnvironment(
            q_values=self.q_values.copy(),
            reward_distribution=self.reward_distribution,
            dynamics=self.dynamics,
        )

    def _validate_action(self, action: int) -> None:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")


@dataclass
class LinearContextualEnvironment:
    """Linear contextual bandit environment.

    Rewards are generated from ``context @ theta[action]`` plus distribution
    noise. A context provider can return either a shared ``(n_features,)``
    vector or per-action ``(n_arms, n_features)`` features.
    """

    theta: FloatArray
    context_provider: Callable[[np.random.Generator], FloatArray]
    reward_distribution: RewardDistribution = field(
        default_factory=lambda: resolve_distribution("gaussian")
    )

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 2 or theta.shape[0] == 0 or theta.shape[1] == 0:
            raise ValueError("theta must have shape (n_arms, n_features)")
        self.theta = theta

    @property
    def n_arms(self) -> int:
        return int(self.theta.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.theta.shape[1])

    def context(self, rng: np.random.Generator) -> FloatArray:
        context = np.asarray(self.context_provider(rng), dtype=float)
        if context.shape == (self.n_features,):
            return np.repeat(context[np.newaxis, :], self.n_arms, axis=0)
        if context.shape == (self.n_arms, self.n_features):
            return context
        raise ValueError(
            "context must have shape (n_features,) or (n_arms, n_features)"
        )

    def expected_rewards(self, context: FloatArray) -> FloatArray:
        if context.shape != (self.n_arms, self.n_features):
            raise ValueError("context shape does not match environment")
        return cast(
            FloatArray, np.einsum("ij,ij->i", context, self.theta).astype(float)
        )

    def step(
        self, action: int, *, context: FloatArray, rng: np.random.Generator
    ) -> float:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")
        rewards = self.expected_rewards(context)
        return self.reward_distribution.sample_one(float(rewards[action]), rng)

    def copy(self) -> LinearContextualEnvironment:
        return LinearContextualEnvironment(
            theta=self.theta.copy(),
            context_provider=self.context_provider,
            reward_distribution=self.reward_distribution,
        )


def make_dynamics(
    change_type: EnvironmentChangeType | str | EnvironmentDynamics,
    params: dict[str, float | int | bool] | None = None,
) -> EnvironmentDynamics:
    """Create a dynamics object from enum/string/config inputs."""

    if isinstance(change_type, EnvironmentDynamics):
        return change_type
    values = {} if params is None else dict(params)
    kind = (
        change_type
        if isinstance(change_type, EnvironmentChangeType)
        else EnvironmentChangeType(str(change_type))
    )
    if kind is EnvironmentChangeType.STATIONARY:
        return StationaryDynamics()
    if kind is EnvironmentChangeType.GRADUAL:
        return GradualDrift(change_rate=float(values.get("change_rate", 0.01)))
    if kind is EnvironmentChangeType.ABRUPT:
        return AbruptShift(
            change_frequency=int(values.get("change_frequency", 100)),
            change_magnitude=float(values.get("change_magnitude", 0.5)),
            shift_at_step_zero=bool(values.get("shift_at_step_zero", False)),
        )
    if kind is EnvironmentChangeType.RANDOM_ARM_SWAPPING:
        return RandomArmSwap(
            shift_probability=float(values.get("shift_probability", 0.2))
        )
    raise ValueError(f"Unknown environment change type: {change_type}")


# Backward-compatible mixin names.
EnvironmentChangeMixin = EnvironmentDynamics
StationaryEnvironmentMixin = StationaryDynamics
GradualChangeEnvironmentMixin = GradualDrift
AbruptChangeEnvironmentMixin = AbruptShift
RandomArmSwappingEnvironmentMixin = RandomArmSwap
