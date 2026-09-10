"""Authoritative execution catalog for every concrete public PyMAB policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from pymab._reference.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    EpsilonGreedyPolicy,
    EXP3Policy,
    GaussianBayesianUCBPolicy,
    GaussianThompsonSamplingPolicy,
    GradientBanditPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
    MedianEliminationPolicy,
    MOSSPolicy,
    PageHinkleyUCBPolicy,
    RandomPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    SuccessiveEliminationPolicy,
    UCBPolicy,
)
from pymab.policies.policy import ContextualPolicy, Policy

PolicyId = Literal[
    "random",
    "greedy",
    "epsilon-greedy",
    "decaying-epsilon-greedy",
    "softmax",
    "gradient-bandit",
    "ucb",
    "kl-ucb",
    "moss",
    "bernoulli-thompson-sampling",
    "gaussian-thompson-sampling",
    "bernoulli-bayesian-ucb",
    "gaussian-bayesian-ucb",
    "sliding-window-ucb",
    "discounted-ucb",
    "sliding-window-bernoulli-thompson-sampling",
    "discounted-bernoulli-thompson-sampling",
    "change-point-ucb",
    "cusum-ucb",
    "page-hinkley-ucb",
    "successive-elimination",
    "median-elimination",
    "exp3",
    "linear-epsilon-greedy",
    "linucb",
    "linear-thompson-sampling",
    "logistic-contextual-bandit",
]
PolicyFamily = Literal[
    "foundations",
    "optimism",
    "bayesian",
    "changing",
    "best-arm",
    "adversarial",
    "contextual",
]
EnvironmentKind = Literal[
    "stationary-bernoulli",
    "stationary-gaussian",
    "changing-bernoulli",
    "best-arm",
    "adversarial",
    "contextual-linear",
    "contextual-logistic",
]
Objective = Literal["cumulative-reward", "best-arm"]


@dataclass(frozen=True)
class PolicySpec:
    """Execution metadata for one stable Arcade policy ID."""

    policy_id: PolicyId
    policy_class: type[Policy] | type[ContextualPolicy]
    family: PolicyFamily
    environment: EnvironmentKind
    objective: Objective
    defaults: dict[str, object]
    guided: dict[str, object]
    seed: int
    challenge_seed: int
    horizon: int = 20
    challenge_horizon: int = 30

    def constructor_parameters(
        self, parameters: dict[str, object], *, horizon: int
    ) -> dict[str, object]:
        """Add fixed game dimensions required by the public constructor."""

        result = dict(parameters)
        result["n_arms"] = 3
        if self.policy_id == "moss":
            result["horizon"] = horizon
        if self.family == "contextual":
            result["n_features"] = 4
        return result

    def create(
        self, parameters: dict[str, object], *, horizon: int
    ) -> Policy | ContextualPolicy:
        """Construct the deterministic Python reference policy."""

        factory = cast(Any, self.policy_class)
        return cast(
            Policy | ContextualPolicy,
            factory(**self.constructor_parameters(parameters, horizon=horizon)),
        )


def _spec(
    policy_id: PolicyId,
    policy_class: type[Policy] | type[ContextualPolicy],
    family: PolicyFamily,
    environment: EnvironmentKind,
    *,
    defaults: dict[str, object],
    guided: dict[str, object] | None = None,
    objective: Objective = "cumulative-reward",
    seed: int,
    challenge_seed: int,
    horizon: int = 20,
    challenge_horizon: int = 30,
) -> PolicySpec:
    return PolicySpec(
        policy_id=policy_id,
        policy_class=policy_class,
        family=family,
        environment=environment,
        objective=objective,
        defaults=defaults,
        guided=dict(defaults if guided is None else guided),
        seed=seed,
        challenge_seed=challenge_seed,
        horizon=horizon,
        challenge_horizon=challenge_horizon,
    )


POLICY_CATALOG: dict[PolicyId, PolicySpec] = {
    spec.policy_id: spec
    for spec in (
        _spec(
            "random",
            RandomPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={},
            seed=101,
            challenge_seed=1101,
        ),
        _spec(
            "greedy",
            GreedyPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={"initial_value": 0.0},
            guided={"initial_value": 0.5},
            seed=102,
            challenge_seed=1102,
        ),
        _spec(
            "epsilon-greedy",
            EpsilonGreedyPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={"initial_value": 0.0, "epsilon": 0.1},
            guided={"initial_value": 0.0, "epsilon": 0.2},
            seed=42,
            challenge_seed=7,
            horizon=12,
            challenge_horizon=20,
        ),
        _spec(
            "decaying-epsilon-greedy",
            DecayingEpsilonGreedyPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={
                "initial_value": 0.0,
                "initial_epsilon": 1.0,
                "min_epsilon": 0.01,
                "decay_rate": 0.01,
            },
            guided={
                "initial_value": 0.0,
                "initial_epsilon": 0.8,
                "min_epsilon": 0.05,
                "decay_rate": 0.2,
            },
            seed=104,
            challenge_seed=1104,
        ),
        _spec(
            "softmax",
            SoftmaxPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={"initial_value": 0.0, "temperature": 1.0},
            guided={"initial_value": 0.0, "temperature": 0.35},
            seed=105,
            challenge_seed=1105,
        ),
        _spec(
            "gradient-bandit",
            GradientBanditPolicy,
            "foundations",
            "stationary-bernoulli",
            defaults={"learning_rate": 0.1, "use_baseline": True},
            seed=106,
            challenge_seed=1106,
        ),
        _spec(
            "ucb",
            UCBPolicy,
            "optimism",
            "stationary-bernoulli",
            defaults={"initial_value": 0.0, "c": 2.0, "reward_scale": 1.0},
            guided={"initial_value": 0.0, "c": 1.0, "reward_scale": 1.0},
            seed=201,
            challenge_seed=1201,
        ),
        _spec(
            "kl-ucb",
            KLUCBPolicy,
            "optimism",
            "stationary-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 3.0,
                "tolerance": 1e-6,
                "max_iterations": 32,
            },
            seed=202,
            challenge_seed=1202,
        ),
        _spec(
            "moss",
            MOSSPolicy,
            "optimism",
            "stationary-bernoulli",
            defaults={"initial_value": 0.0, "c": 1.0, "reward_scale": 1.0},
            seed=203,
            challenge_seed=1203,
        ),
        _spec(
            "bernoulli-thompson-sampling",
            BernoulliThompsonSamplingPolicy,
            "bayesian",
            "stationary-bernoulli",
            defaults={"alpha_prior": 1.0, "beta_prior": 1.0},
            seed=301,
            challenge_seed=1301,
        ),
        _spec(
            "gaussian-thompson-sampling",
            GaussianThompsonSamplingPolicy,
            "bayesian",
            "stationary-gaussian",
            defaults={
                "prior_mean": 0.0,
                "prior_precision": 1.0,
                "reward_precision": 1.0,
            },
            seed=302,
            challenge_seed=1302,
        ),
        _spec(
            "bernoulli-bayesian-ucb",
            BernoulliBayesianUCBPolicy,
            "bayesian",
            "stationary-bernoulli",
            defaults={"alpha_prior": 1.0, "beta_prior": 1.0, "quantile": 0.95},
            seed=303,
            challenge_seed=1303,
        ),
        _spec(
            "gaussian-bayesian-ucb",
            GaussianBayesianUCBPolicy,
            "bayesian",
            "stationary-gaussian",
            defaults={
                "prior_mean": 0.0,
                "prior_precision": 1.0,
                "reward_precision": 1.0,
                "quantile": 0.95,
            },
            seed=304,
            challenge_seed=1304,
        ),
        _spec(
            "sliding-window-ucb",
            SlidingWindowUCBPolicy,
            "changing",
            "changing-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 2.0,
                "reward_scale": 1.0,
                "window_size": 100,
            },
            guided={
                "initial_value": 0.0,
                "c": 1.0,
                "reward_scale": 1.0,
                "window_size": 8,
            },
            seed=401,
            challenge_seed=1401,
            horizon=30,
            challenge_horizon=40,
        ),
        _spec(
            "discounted-ucb",
            DiscountedUCBPolicy,
            "changing",
            "changing-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 2.0,
                "reward_scale": 1.0,
                "discount_factor": 0.9,
            },
            seed=402,
            challenge_seed=1402,
            horizon=30,
            challenge_horizon=40,
        ),
        _spec(
            "sliding-window-bernoulli-thompson-sampling",
            SlidingWindowBernoulliThompsonSamplingPolicy,
            "changing",
            "changing-bernoulli",
            defaults={"alpha_prior": 1.0, "beta_prior": 1.0, "window_size": 100},
            guided={"alpha_prior": 1.0, "beta_prior": 1.0, "window_size": 8},
            seed=403,
            challenge_seed=1403,
            horizon=30,
            challenge_horizon=40,
        ),
        _spec(
            "discounted-bernoulli-thompson-sampling",
            DiscountedBernoulliThompsonSamplingPolicy,
            "changing",
            "changing-bernoulli",
            defaults={"alpha_prior": 1.0, "beta_prior": 1.0, "discount_factor": 0.95},
            seed=404,
            challenge_seed=1404,
            horizon=30,
            challenge_horizon=40,
        ),
        _spec(
            "change-point-ucb",
            ChangePointUCBPolicy,
            "changing",
            "changing-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 2.0,
                "reward_scale": 1.0,
                "detector": "cusum",
                "threshold": 5.0,
                "drift": 0.05,
                "min_observations": 20,
            },
            guided={
                "initial_value": 0.0,
                "c": 1.0,
                "reward_scale": 1.0,
                "detector": "cusum",
                "threshold": 1.0,
                "drift": 0.02,
                "min_observations": 4,
            },
            seed=405,
            challenge_seed=1405,
            horizon=36,
            challenge_horizon=45,
        ),
        _spec(
            "cusum-ucb",
            CUSUMUCBPolicy,
            "changing",
            "changing-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 2.0,
                "reward_scale": 1.0,
                "threshold": 5.0,
                "drift": 0.05,
                "min_observations": 20,
            },
            guided={
                "initial_value": 0.0,
                "c": 1.0,
                "reward_scale": 1.0,
                "threshold": 1.0,
                "drift": 0.02,
                "min_observations": 4,
            },
            seed=406,
            challenge_seed=1406,
            horizon=36,
            challenge_horizon=45,
        ),
        _spec(
            "page-hinkley-ucb",
            PageHinkleyUCBPolicy,
            "changing",
            "changing-bernoulli",
            defaults={
                "initial_value": 0.0,
                "c": 2.0,
                "reward_scale": 1.0,
                "threshold": 5.0,
                "drift": 0.05,
                "min_observations": 20,
            },
            guided={
                "initial_value": 0.0,
                "c": 1.0,
                "reward_scale": 1.0,
                "threshold": 1.0,
                "drift": 0.02,
                "min_observations": 4,
            },
            seed=407,
            challenge_seed=1407,
            horizon=36,
            challenge_horizon=45,
        ),
        _spec(
            "successive-elimination",
            SuccessiveEliminationPolicy,
            "best-arm",
            "best-arm",
            defaults={"delta": 0.05, "confidence_scale": 1.0},
            guided={"delta": 0.2, "confidence_scale": 0.3},
            objective="best-arm",
            seed=501,
            challenge_seed=1501,
            horizon=60,
            challenge_horizon=80,
        ),
        _spec(
            "median-elimination",
            MedianEliminationPolicy,
            "best-arm",
            "best-arm",
            defaults={"epsilon": 0.1, "delta": 0.05},
            guided={"epsilon": 0.8, "delta": 0.5},
            objective="best-arm",
            seed=502,
            challenge_seed=1502,
            horizon=60,
            challenge_horizon=80,
        ),
        _spec(
            "exp3",
            EXP3Policy,
            "adversarial",
            "adversarial",
            defaults={"gamma": 0.07, "learning_rate": None},
            guided={"gamma": 0.2, "learning_rate": None},
            seed=601,
            challenge_seed=1601,
            horizon=30,
            challenge_horizon=40,
        ),
        _spec(
            "linear-epsilon-greedy",
            LinearEpsilonGreedyPolicy,
            "contextual",
            "contextual-linear",
            defaults={"epsilon": 0.1, "learning_rate": 0.1},
            seed=701,
            challenge_seed=1701,
        ),
        _spec(
            "linucb",
            LinUCBPolicy,
            "contextual",
            "contextual-logistic",
            defaults={"alpha": 1.0, "l2": 1.0},
            seed=31415,
            challenge_seed=20260824,
            horizon=12,
            challenge_horizon=20,
        ),
        _spec(
            "linear-thompson-sampling",
            LinearThompsonSamplingPolicy,
            "contextual",
            "contextual-linear",
            defaults={"exploration_scale": 1.0, "l2": 1.0},
            seed=703,
            challenge_seed=1703,
        ),
        _spec(
            "logistic-contextual-bandit",
            LogisticContextualBanditPolicy,
            "contextual",
            "contextual-logistic",
            defaults={"epsilon": 0.05, "learning_rate": 0.1, "l2": 0.0},
            seed=704,
            challenge_seed=1704,
        ),
    )
}


def public_policy_classes() -> set[str]:
    """Return class names represented by the execution catalog."""

    return {spec.policy_class.__name__ for spec in POLICY_CATALOG.values()}


def catalog_payload() -> list[dict[str, Any]]:
    """Return JSON-safe catalog fields used by cross-language tests."""

    return [
        {
            "policyId": spec.policy_id,
            "className": spec.policy_class.__name__,
            "family": spec.family,
            "environment": spec.environment,
            "objective": spec.objective,
            "defaults": spec.defaults,
            "guided": spec.guided,
            "seed": spec.seed,
            "challengeSeed": spec.challenge_seed,
            "horizon": spec.horizon,
            "challengeHorizon": spec.challenge_horizon,
        }
        for spec in POLICY_CATALOG.values()
    ]


__all__ = [
    "POLICY_CATALOG",
    "EnvironmentKind",
    "Objective",
    "PolicyFamily",
    "PolicyId",
    "PolicySpec",
    "catalog_payload",
    "public_policy_classes",
]
