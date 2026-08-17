"""Bandit environments and support-aware non-stationary dynamics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

import numpy as np

from pymab.distributions import (
    ArmPrior,
    BernoulliReward,
    GaussianReward,
    RewardModel,
)
from pymab.types import FloatArray, RewardDomain


class EnvironmentDynamics(ABC):
    """Strategy for evolving true arm means."""

    supported_domains: frozenset[RewardDomain]

    @abstractmethod
    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        """Return the arm means for the supplied step."""

    def supports(self, domain: RewardDomain) -> bool:
        return domain in self.supported_domains

    def clone(self) -> EnvironmentDynamics:
        """Return independent dynamics state for one replicate."""

        return deepcopy(self)


class ContextProvider(ABC):
    """Cloneable source of contextual feature matrices."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Return context features for one decision."""

    def clone(self) -> ContextProvider:
        """Return independent provider state for one replicate."""

        return deepcopy(self)


@dataclass(frozen=True)
class CallableContextProvider(ContextProvider):
    """Stateless adapter for a context-producing callable."""

    function: Callable[[np.random.Generator], FloatArray]

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return self.function(rng)

    def clone(self) -> CallableContextProvider:
        return self


@runtime_checkable
class ClassicEnvironment(Protocol):
    """Structural contract for a classic non-contextual environment."""

    @property
    def n_arms(self) -> int: ...

    @property
    def reward_model(self) -> RewardModel: ...

    @property
    def reward_domain(self) -> RewardDomain: ...

    @property
    def contextual(self) -> bool: ...

    def clone(self) -> ClassicEnvironment: ...

    def advance(self, *, step: int, rng: np.random.Generator) -> None: ...

    def expected_rewards(self) -> FloatArray: ...


@runtime_checkable
class ContextualEnvironment(Protocol):
    """Structural contract for a contextual environment."""

    @property
    def n_arms(self) -> int: ...

    @property
    def reward_model(self) -> RewardModel: ...

    @property
    def n_features(self) -> int: ...

    @property
    def reward_domain(self) -> RewardDomain: ...

    @property
    def contextual(self) -> bool: ...

    def clone(self) -> ContextualEnvironment: ...

    def context(self, rng: np.random.Generator) -> FloatArray: ...

    def expected_rewards(self, context: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class StationaryDynamics(EnvironmentDynamics):
    supported_domains = frozenset(RewardDomain)

    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        return means


@dataclass(frozen=True)
class GradualDrift(EnvironmentDynamics):
    """Gaussian random-walk drift for unbounded real-valued means."""

    std: float = 0.01
    supported_domains = frozenset({RewardDomain.REAL})

    def __post_init__(self) -> None:
        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("std must be finite and non-negative")

    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        return np.asarray(means + rng.normal(0.0, self.std, size=means.shape))


@dataclass(frozen=True)
class AbruptShift(EnvironmentDynamics):
    """Periodic Gaussian shifts for unbounded real-valued means."""

    frequency: int = 100
    std: float = 0.5
    shift_at_step_zero: bool = False
    supported_domains = frozenset({RewardDomain.REAL})

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, int) or isinstance(self.frequency, bool):
            raise TypeError("frequency must be an integer")
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("std must be finite and non-negative")

    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        if step == 0 and not self.shift_at_step_zero:
            return means
        if step % self.frequency == 0:
            return np.asarray(means + rng.normal(0.0, self.std, size=means.shape))
        return means


@dataclass(frozen=True)
class ProbabilityDrift(EnvironmentDynamics):
    """Gaussian random walk in log-odds space for probability means."""

    logit_std: float = 0.05
    epsilon: float = 1e-9
    supported_domains = frozenset({RewardDomain.BINARY, RewardDomain.UNIT_INTERVAL})

    def __post_init__(self) -> None:
        if not np.isfinite(self.logit_std) or self.logit_std < 0:
            raise ValueError("logit_std must be finite and non-negative")
        if not np.isfinite(self.epsilon) or not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")

    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        clipped = np.clip(means, self.epsilon, 1.0 - self.epsilon)
        logits = np.log(clipped / (1.0 - clipped))
        logits += rng.normal(0.0, self.logit_std, size=means.shape)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=float)


@dataclass(frozen=True)
class RandomArmSwap(EnvironmentDynamics):
    probability: float = 0.2
    supported_domains = frozenset(RewardDomain)

    def __post_init__(self) -> None:
        if not np.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")

    def apply(
        self, means: FloatArray, *, step: int, rng: np.random.Generator
    ) -> FloatArray:
        if rng.random() < self.probability:
            return np.asarray(rng.permutation(means), dtype=float)
        return means


@dataclass(eq=False)
class BanditEnvironment:
    """Classic K-armed simulator with support-aware rewards and dynamics."""

    means: FloatArray
    reward_model: RewardModel = field(default_factory=GaussianReward)
    dynamics: EnvironmentDynamics = field(default_factory=StationaryDynamics)

    def __post_init__(self) -> None:
        values = np.asarray(self.means, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("means must be a non-empty 1D array")
        self.reward_model.validate_means(values)
        if not self.dynamics.supports(self.reward_model.domain):
            raise ValueError(
                f"{type(self.dynamics).__name__} does not support "
                f"{self.reward_model.domain.value} rewards"
            )
        self.means = values.copy()

    @classmethod
    def from_prior(
        cls,
        *,
        n_arms: int,
        prior: ArmPrior,
        reward_model: RewardModel | None = None,
        dynamics: EnvironmentDynamics | None = None,
        rng: np.random.Generator,
    ) -> BanditEnvironment:
        model = GaussianReward() if reward_model is None else reward_model
        means = prior.sample(n_arms=n_arms, rng=rng)
        model.validate_means(means)
        return cls(
            means=means,
            reward_model=model,
            dynamics=StationaryDynamics() if dynamics is None else dynamics,
        )

    @property
    def n_arms(self) -> int:
        return int(self.means.size)

    @property
    def reward_domain(self) -> RewardDomain:
        return self.reward_model.domain

    @property
    def contextual(self) -> bool:
        return False

    def expected_rewards(self) -> FloatArray:
        return self.means.copy()

    def sample_rewards(self, *, rng: np.random.Generator) -> FloatArray:
        return self.reward_model.sample(self.means, rng)

    def advance(self, *, step: int, rng: np.random.Generator) -> None:
        values = np.asarray(self.dynamics.apply(self.means, step=step, rng=rng))
        if values.shape != self.means.shape:
            raise ValueError("dynamics must preserve the arm-mean shape")
        self.reward_model.validate_means(values)
        self.means = values.copy()

    def clone(self) -> BanditEnvironment:
        return BanditEnvironment(
            means=self.means.copy(),
            reward_model=self.reward_model.clone(),
            dynamics=self.dynamics.clone(),
        )


@dataclass(eq=False)
class LinearContextualEnvironment:
    """Contextual simulator whose expected rewards are linear in features."""

    theta: FloatArray
    context_provider: ContextProvider | Callable[[np.random.Generator], FloatArray]
    reward_model: RewardModel = field(default_factory=GaussianReward)

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 2 or 0 in theta.shape or not np.all(np.isfinite(theta)):
            raise ValueError("theta must be a finite (n_arms, n_features) matrix")
        if self.reward_model.domain is RewardDomain.BINARY:
            raise ValueError("use LogisticContextualEnvironment for Bernoulli rewards")
        self.theta = theta.copy()
        if not isinstance(self.context_provider, ContextProvider):
            self.context_provider = CallableContextProvider(self.context_provider)

    @property
    def n_arms(self) -> int:
        return int(self.theta.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.theta.shape[1])

    @property
    def reward_domain(self) -> RewardDomain:
        return self.reward_model.domain

    @property
    def contextual(self) -> bool:
        return True

    def context(self, rng: np.random.Generator) -> FloatArray:
        provider = cast(ContextProvider, self.context_provider)
        value = np.asarray(provider.sample(rng), dtype=float)
        if value.shape == (self.n_features,):
            value = np.repeat(value[np.newaxis, :], self.n_arms, axis=0)
        if value.shape != (self.n_arms, self.n_features):
            raise ValueError(
                "context must have shape (n_features,) or (n_arms, n_features)"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("context must contain only finite values")
        return value

    def expected_rewards(self, context: FloatArray) -> FloatArray:
        self._validate_context(context)
        means = np.einsum("ij,ij->i", context, self.theta).astype(float)
        self.reward_model.validate_means(means)
        return np.asarray(means)

    def sample_rewards(
        self, *, context: FloatArray, rng: np.random.Generator
    ) -> FloatArray:
        return self.reward_model.sample(self.expected_rewards(context), rng)

    def clone(self) -> LinearContextualEnvironment:
        return type(self)(
            theta=self.theta.copy(),
            context_provider=cast(ContextProvider, self.context_provider).clone(),
            reward_model=self.reward_model.clone(),
        )

    def _validate_context(self, context: FloatArray) -> None:
        if context.shape != (self.n_arms, self.n_features):
            raise ValueError("context shape does not match environment")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")


@dataclass(eq=False)
class LogisticContextualEnvironment(LinearContextualEnvironment):
    """Contextual Bernoulli simulator using a logistic link."""

    reward_model: RewardModel = field(default_factory=BernoulliReward)

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 2 or 0 in theta.shape or not np.all(np.isfinite(theta)):
            raise ValueError("theta must be a finite (n_arms, n_features) matrix")
        if self.reward_model.domain is not RewardDomain.BINARY:
            raise ValueError("logistic environments require a binary reward model")
        self.theta = theta.copy()
        if not isinstance(self.context_provider, ContextProvider):
            self.context_provider = CallableContextProvider(self.context_provider)

    def expected_rewards(self, context: FloatArray) -> FloatArray:
        self._validate_context(context)
        logits = np.einsum("ij,ij->i", context, self.theta).astype(float)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        return np.asarray(probabilities, dtype=float)


Environment = ClassicEnvironment | ContextualEnvironment


__all__ = [
    "AbruptShift",
    "BanditEnvironment",
    "CallableContextProvider",
    "ClassicEnvironment",
    "ContextProvider",
    "ContextualEnvironment",
    "Environment",
    "EnvironmentDynamics",
    "GradualDrift",
    "LinearContextualEnvironment",
    "LogisticContextualEnvironment",
    "ProbabilityDrift",
    "RandomArmSwap",
    "StationaryDynamics",
]
