"""Canonical workloads shared by Python and Rust backend benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import (
    BanditEnvironment,
    GaussianContextProvider,
    LogisticContextualEnvironment,
    ProbabilityDrift,
)
from pymab.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    CUSUMUCBPolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    EpsilonGreedyPolicy,
    EXP3Policy,
    GaussianThompsonSamplingPolicy,
    GradientBanditPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    UCBPolicy,
)
from pymab.simulation import BackendMode, Experiment, ExperimentConfig

CASE_NAMES = ("stationary", "bernoulli", "nonstationary", "contextual")


@dataclass(frozen=True)
class CaseDefaults:
    """Default scale and seed for one canonical benchmark workload."""

    horizon: int
    n_replicates: int
    seed: int


_CASE_DEFAULTS = {
    "stationary": CaseDefaults(horizon=2_000, n_replicates=64, seed=101),
    "bernoulli": CaseDefaults(horizon=2_000, n_replicates=64, seed=202),
    "nonstationary": CaseDefaults(horizon=2_000, n_replicates=64, seed=303),
    "contextual": CaseDefaults(horizon=1_000, n_replicates=32, seed=404),
}


def case_defaults(name: str) -> CaseDefaults:
    """Return defaults for a registered benchmark case.

    Raises:
        ValueError: If ``name`` is not a registered case.
    """

    try:
        return _CASE_DEFAULTS[name]
    except KeyError as exc:
        choices = ", ".join(CASE_NAMES)
        raise ValueError(
            f"unknown benchmark case {name!r}; choose from {choices}"
        ) from exc


def build_experiment(
    name: str,
    *,
    horizon: int,
    n_replicates: int,
    backend: BackendMode = "auto",
) -> Experiment:
    """Construct a fresh canonical experiment at the requested scale.

    Args:
        name: Registered benchmark case name.
        horizon: Number of decisions per replicate.
        n_replicates: Number of independent replicates.

    Returns:
        A new experiment whose policy state has not been updated.

    Raises:
        ValueError: If ``name`` is not registered.
    """

    defaults = case_defaults(name)
    builders: dict[str, Callable[[ExperimentConfig], Experiment]] = {
        "stationary": _stationary_experiment,
        "bernoulli": _bernoulli_experiment,
        "nonstationary": _nonstationary_experiment,
        "contextual": _contextual_experiment,
    }
    config = ExperimentConfig(
        horizon=horizon,
        n_replicates=n_replicates,
        seed=defaults.seed,
        backend=backend,
    )
    return builders[name](config)


def _stationary_experiment(config: ExperimentConfig) -> Experiment:
    n_arms = 8
    return Experiment(
        environment=BanditEnvironment(
            means=np.linspace(-0.25, 1.0, n_arms),
            reward_model=GaussianReward(std=1.0),
        ),
        policies={
            "greedy": GreedyPolicy(n_arms=n_arms),
            "epsilon": EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1),
            "decaying": DecayingEpsilonGreedyPolicy(n_arms=n_arms),
            "softmax": SoftmaxPolicy(n_arms=n_arms, temperature=0.5),
            "ucb": UCBPolicy(n_arms=n_arms),
            "gradient": GradientBanditPolicy(n_arms=n_arms),
            "thompson": GaussianThompsonSamplingPolicy(n_arms=n_arms),
        },
        config=config,
    )


def _bernoulli_experiment(config: ExperimentConfig) -> Experiment:
    n_arms = 8
    return Experiment(
        environment=BanditEnvironment(
            means=np.linspace(0.1, 0.8, n_arms),
            reward_model=BernoulliReward(),
        ),
        policies={
            "thompson": BernoulliThompsonSamplingPolicy(n_arms=n_arms),
            "bayesian-ucb": BernoulliBayesianUCBPolicy(n_arms=n_arms),
            "kl-ucb": KLUCBPolicy(n_arms=n_arms),
            "exp3": EXP3Policy(n_arms=n_arms),
        },
        config=config,
    )


def _nonstationary_experiment(config: ExperimentConfig) -> Experiment:
    n_arms = 8
    return Experiment(
        environment=BanditEnvironment(
            means=np.linspace(0.1, 0.8, n_arms),
            reward_model=BernoulliReward(),
            dynamics=ProbabilityDrift(logit_std=0.02),
        ),
        policies={
            "sliding-ucb": SlidingWindowUCBPolicy(n_arms=n_arms, window_size=200),
            "discounted-ucb": DiscountedUCBPolicy(n_arms=n_arms, discount_factor=0.99),
            "sliding-thompson": SlidingWindowBernoulliThompsonSamplingPolicy(
                n_arms=n_arms, window_size=200
            ),
            "discounted-thompson": DiscountedBernoulliThompsonSamplingPolicy(
                n_arms=n_arms, discount_factor=0.99
            ),
            "cusum-ucb": CUSUMUCBPolicy(
                n_arms=n_arms,
                threshold=2.0,
                drift=0.02,
                min_observations=20,
            ),
        },
        config=config,
    )


def _contextual_experiment(config: ExperimentConfig) -> Experiment:
    n_arms = 4
    n_features = 8
    theta = np.array(
        [
            [0.8, 0.2, -0.1, 0.0, 0.1, 0.0, 0.2, -0.2],
            [0.0, 0.7, 0.2, -0.1, 0.0, 0.1, -0.2, 0.2],
            [-0.2, 0.0, 0.8, 0.2, -0.1, 0.0, 0.1, 0.1],
            [0.1, -0.2, 0.0, 0.7, 0.2, -0.1, 0.0, 0.2],
        ],
        dtype=float,
    )
    return Experiment(
        environment=LogisticContextualEnvironment(
            theta=theta,
            context_provider=GaussianContextProvider(
                n_arms=n_arms,
                n_features=n_features,
            ),
            reward_model=BernoulliReward(),
        ),
        policies={
            "linear-epsilon": LinearEpsilonGreedyPolicy(
                n_arms=n_arms,
                n_features=n_features,
                epsilon=0.1,
            ),
            "linucb": LinUCBPolicy(n_arms=n_arms, n_features=n_features),
            "linear-thompson": LinearThompsonSamplingPolicy(
                n_arms=n_arms,
                n_features=n_features,
            ),
            "logistic": LogisticContextualBanditPolicy(
                n_arms=n_arms,
                n_features=n_features,
                epsilon=0.05,
            ),
        },
        config=config,
    )


__all__ = ["CASE_NAMES", "CaseDefaults", "build_experiment", "case_defaults"]
