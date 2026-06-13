"""Reward distributions for bandit environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class RewardDistribution(ABC):
    """Base class for vectorized reward distributions."""

    @abstractmethod
    def sample(self, q_values: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Sample one reward per supplied arm value."""

    @abstractmethod
    def initial_values(
        self, *, mean: float, scale: float, n_arms: int, rng: np.random.Generator
    ) -> FloatArray:
        """Generate initial true arm values."""

    def sample_one(
        self, q_value: float, rng: np.random.Generator | None = None
    ) -> float:
        """Sample a scalar reward for a single arm."""

        generator = np.random.default_rng() if rng is None else rng
        return float(self.sample(np.array([q_value], dtype=float), generator)[0])

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        """Compatibility hook implemented by concrete classes."""

        raise NotImplementedError

    @staticmethod
    def generate_Q_values(q_value: float, variance: float, size: int) -> FloatArray:
        """Compatibility hook implemented by concrete classes."""

        raise NotImplementedError


@dataclass(frozen=True)
class GaussianReward(RewardDistribution):
    """Gaussian rewards with known observation standard deviation."""

    std: float = 1.0

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("std must be positive")

    def sample(self, q_values: FloatArray, rng: np.random.Generator) -> FloatArray:
        return rng.normal(loc=q_values, scale=self.std).astype(float)

    def initial_values(
        self, *, mean: float, scale: float, n_arms: int, rng: np.random.Generator
    ) -> FloatArray:
        _validate_n_arms(n_arms)
        if scale < 0:
            raise ValueError("scale must be non-negative")
        return rng.normal(loc=mean, scale=scale, size=n_arms).astype(float)

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        return float(np.random.default_rng().normal(q_value, variance))

    @staticmethod
    def generate_Q_values(q_value: float, variance: float, size: int) -> FloatArray:
        return np.random.default_rng().normal(q_value, variance, size).astype(float)


@dataclass(frozen=True)
class BernoulliReward(RewardDistribution):
    """Bernoulli rewards where each arm value is a success probability."""

    def sample(self, q_values: FloatArray, rng: np.random.Generator) -> FloatArray:
        self._validate_probabilities(q_values)
        return rng.binomial(1, q_values).astype(float)

    def initial_values(
        self, *, mean: float, scale: float, n_arms: int, rng: np.random.Generator
    ) -> FloatArray:
        _validate_n_arms(n_arms)
        if not 0 <= mean <= 1:
            raise ValueError("mean must be in [0, 1] for Bernoulli rewards")
        concentration = max(scale, 2.0)
        alpha = max(mean * concentration, 1e-6)
        beta = max((1 - mean) * concentration, 1e-6)
        return rng.beta(alpha, beta, size=n_arms).astype(float)

    @staticmethod
    def _validate_probabilities(q_values: FloatArray) -> None:
        if np.any((q_values < 0) | (q_values > 1)):
            raise ValueError("Bernoulli arm probabilities must be in [0, 1]")

    @staticmethod
    def get_reward(q_value: float, variance: float = 0.0) -> float:
        if not 0 <= q_value <= 1:
            raise ValueError("Bernoulli probability must be in [0, 1]")
        return float(np.random.default_rng().binomial(1, q_value))

    @staticmethod
    def generate_Q_values(q_value: float, variance: float, size: int) -> FloatArray:
        if not 0 <= q_value <= 1:
            raise ValueError("Bernoulli probability must be in [0, 1]")
        concentration = max(variance, 2.0)
        return (
            np.random.default_rng()
            .beta(
                max(q_value * concentration, 1e-6),
                max((1 - q_value) * concentration, 1e-6),
                size=size,
            )
            .astype(float)
        )


@dataclass(frozen=True)
class UniformReward(RewardDistribution):
    """Uniform rewards centered on the true arm value."""

    half_width: float = 1.0

    def __post_init__(self) -> None:
        if self.half_width < 0:
            raise ValueError("half_width must be non-negative")

    def sample(self, q_values: FloatArray, rng: np.random.Generator) -> FloatArray:
        return rng.uniform(q_values - self.half_width, q_values + self.half_width)

    def initial_values(
        self, *, mean: float, scale: float, n_arms: int, rng: np.random.Generator
    ) -> FloatArray:
        _validate_n_arms(n_arms)
        if scale < 0:
            raise ValueError("scale must be non-negative")
        return rng.uniform(mean - scale, mean + scale, size=n_arms).astype(float)

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        return float(
            np.random.default_rng().uniform(q_value - variance, q_value + variance)
        )

    @staticmethod
    def generate_Q_values(q_value: float, variance: float, size: int) -> FloatArray:
        return np.random.default_rng().uniform(
            q_value - variance, q_value + variance, size
        )


def resolve_distribution(
    distribution: str | RewardDistribution | type[RewardDistribution],
    *,
    reward_scale: float = 1.0,
) -> RewardDistribution:
    """Resolve strings/classes/instances into a distribution instance."""

    if isinstance(distribution, RewardDistribution):
        return distribution
    if isinstance(distribution, type) and issubclass(distribution, RewardDistribution):
        return distribution()
    names = {
        "gaussian": GaussianReward(std=reward_scale),
        "normal": GaussianReward(std=reward_scale),
        "bernoulli": BernoulliReward(),
        "uniform": UniformReward(half_width=reward_scale),
    }
    try:
        return names[str(distribution).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown reward distribution: {distribution}") from exc


def _validate_n_arms(n_arms: int) -> None:
    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
