"""Reward models and explicit priors for simulated arm means."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from pymab.types import FloatArray, RewardDomain


class RewardModel(ABC):
    """Distribution of observed rewards conditional on their true means."""

    domain: RewardDomain

    @abstractmethod
    def sample(self, means: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Sample one potential reward for every supplied arm mean."""

    def sample_one(self, mean: float, *, rng: np.random.Generator) -> float:
        """Sample a scalar reward for one arm using an explicit random stream."""

        return float(self.sample(np.array([mean], dtype=float), rng)[0])

    def clone(self) -> RewardModel:
        """Return an independent reward-model instance for one replicate."""

        return deepcopy(self)

    def validate_means(self, means: FloatArray) -> None:
        """Validate true arm means against this model's support."""

        values = np.asarray(means, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("reward means must be non-empty and finite")


@dataclass(frozen=True)
class GaussianReward(RewardModel):
    """Gaussian observations with known standard deviation."""

    std: float = 1.0
    domain = RewardDomain.REAL

    def __post_init__(self) -> None:
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be finite and positive")

    def sample(self, means: FloatArray, rng: np.random.Generator) -> FloatArray:
        self.validate_means(means)
        return np.asarray(rng.normal(loc=means, scale=self.std), dtype=float)


@dataclass(frozen=True)
class BernoulliReward(RewardModel):
    """Binary observations whose means are success probabilities."""

    domain = RewardDomain.BINARY

    def validate_means(self, means: FloatArray) -> None:
        super().validate_means(means)
        if np.any((means < 0) | (means > 1)):
            raise ValueError("Bernoulli arm probabilities must be in [0, 1]")

    def sample(self, means: FloatArray, rng: np.random.Generator) -> FloatArray:
        self.validate_means(means)
        return np.asarray(rng.binomial(1, means), dtype=float)


@dataclass(frozen=True)
class UniformReward(RewardModel):
    """Uniform observations centered on each true arm mean."""

    half_width: float = 1.0
    domain = RewardDomain.REAL

    def __post_init__(self) -> None:
        if not np.isfinite(self.half_width) or self.half_width < 0:
            raise ValueError("half_width must be finite and non-negative")

    def sample(self, means: FloatArray, rng: np.random.Generator) -> FloatArray:
        self.validate_means(means)
        return np.asarray(
            rng.uniform(means - self.half_width, means + self.half_width),
            dtype=float,
        )


class ArmPrior(ABC):
    """Distribution used only to generate initial true arm means."""

    @abstractmethod
    def sample(self, *, n_arms: int, rng: np.random.Generator) -> FloatArray:
        """Generate one true mean per arm."""


@dataclass(frozen=True)
class GaussianArmPrior(ArmPrior):
    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("std must be finite and non-negative")

    def sample(self, *, n_arms: int, rng: np.random.Generator) -> FloatArray:
        _validate_n_arms(n_arms)
        return np.asarray(rng.normal(self.mean, self.std, size=n_arms), dtype=float)


@dataclass(frozen=True)
class BetaArmPrior(ArmPrior):
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.alpha)
            or not np.isfinite(self.beta)
            or self.alpha <= 0
            or self.beta <= 0
        ):
            raise ValueError("alpha and beta must be finite and positive")

    def sample(self, *, n_arms: int, rng: np.random.Generator) -> FloatArray:
        _validate_n_arms(n_arms)
        return np.asarray(rng.beta(self.alpha, self.beta, size=n_arms), dtype=float)


@dataclass(frozen=True)
class UniformArmPrior(ArmPrior):
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("low and high must be finite")
        if self.low > self.high:
            raise ValueError("low must be <= high")

    def sample(self, *, n_arms: int, rng: np.random.Generator) -> FloatArray:
        _validate_n_arms(n_arms)
        return np.asarray(rng.uniform(self.low, self.high, size=n_arms), dtype=float)


def resolve_reward_model(
    model: str | RewardModel | type[RewardModel], *, observation_scale: float = 1.0
) -> RewardModel:
    """Resolve a short name, class, or instance into a reward model."""

    if isinstance(model, RewardModel):
        return model
    if isinstance(model, type) and issubclass(model, RewardModel):
        return model()
    name = str(model).lower()
    if name in {"gaussian", "normal"}:
        return GaussianReward(std=observation_scale)
    if name == "bernoulli":
        return BernoulliReward()
    if name == "uniform":
        return UniformReward(half_width=observation_scale)
    raise ValueError(f"unknown reward model: {model}")


def _validate_n_arms(n_arms: int) -> None:
    if not isinstance(n_arms, int) or isinstance(n_arms, bool):
        raise TypeError("n_arms must be an integer")
    if n_arms <= 0:
        raise ValueError("n_arms must be positive")


__all__ = [
    "ArmPrior",
    "BernoulliReward",
    "BetaArmPrior",
    "GaussianArmPrior",
    "GaussianReward",
    "RewardModel",
    "UniformArmPrior",
    "UniformReward",
    "resolve_reward_model",
]
