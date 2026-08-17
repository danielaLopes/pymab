"""Real-world PyMAB scenario templates.

Run with:

    python examples/real_world_scenarios.py
"""

from __future__ import annotations

import numpy as np

from pymab import compare
from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import (
    AbruptShift,
    BanditEnvironment,
    GradualDrift,
    LinearContextualEnvironment,
)
from pymab.policies import (
    BernoulliThompsonSamplingPolicy,
    LinUCBPolicy,
    RandomPolicy,
    SlidingWindowUCBPolicy,
    UCBPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig


def recommendation_clicks() -> list[dict[str, str | float | int]]:
    """Choose among recommendation modules using click-through rates."""

    return compare(
        {
            "random": RandomPolicy(n_arms=4),
            "thompson": BernoulliThompsonSamplingPolicy(n_arms=4),
            "ucb": UCBPolicy(n_arms=4),
        },
        environment=BanditEnvironment(
            means=np.array([0.04, 0.06, 0.08, 0.12]),
            reward_model=BernoulliReward(),
        ),
        config=ExperimentConfig(horizon=300, n_replicates=80, seed=1),
    ).summary()


def ad_allocation_by_segment() -> float:
    """Allocate ads when the best creative depends on user segment."""

    def context_provider(rng: np.random.Generator) -> np.ndarray:
        is_mobile = float(rng.random() < 0.55)
        context = np.array([1.0, is_mobile, 1.0 - is_mobile])
        return np.repeat(context[np.newaxis, :], 2, axis=0)

    result = Experiment(
        environment=LinearContextualEnvironment(
            theta=np.array(
                [
                    [0.02, 0.08, 0.0],
                    [0.02, 0.0, 0.08],
                ]
            ),
            context_provider=context_provider,
            reward_model=GaussianReward(std=0.005),
        ),
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=3)},
        config=ExperimentConfig(n_replicates=60, horizon=250, seed=10),
    ).run()
    return float(result.optimal_action_rate_by_step[-50:, 0].mean())


def pricing_experiment() -> list[dict[str, str | float | int]]:
    """Choose among price points using revenue-like continuous rewards."""

    return compare(
        {"random": RandomPolicy(n_arms=4), "ucb": UCBPolicy(n_arms=4)},
        environment=BanditEnvironment(
            means=np.array([7.2, 8.5, 8.1, 6.3]),
            reward_model=GaussianReward(std=0.8),
        ),
        config=ExperimentConfig(horizon=200, n_replicates=100, seed=11),
    ).summary()


def clinical_trial_bernoulli() -> list[dict[str, str | float | int]]:
    """Compare treatment arms with binary success outcomes."""

    return compare(
        {
            "random": RandomPolicy(n_arms=3),
            "thompson": BernoulliThompsonSamplingPolicy(n_arms=3),
        },
        environment=BanditEnvironment(
            means=np.array([0.42, 0.50, 0.58]),
            reward_model=BernoulliReward(),
        ),
        config=ExperimentConfig(horizon=120, n_replicates=120, seed=21),
    ).summary()


def proxy_server_selection() -> list[dict[str, str | float | int]]:
    """Route requests to proxies or servers using latency-savings rewards."""

    return compare(
        {"random": RandomPolicy(n_arms=5), "ucb": UCBPolicy(n_arms=5)},
        environment=BanditEnvironment(
            means=np.array([18.0, 22.0, 15.0, 30.0, 24.0]),
            reward_model=GaussianReward(std=4.0),
        ),
        config=ExperimentConfig(horizon=250, n_replicates=80, seed=31),
    ).summary()


def non_stationary_demand() -> list[dict[str, str | float | int]]:
    """React when demand changes after a launch or seasonal event."""

    return compare(
        {
            "ucb": UCBPolicy(n_arms=3),
            "sliding-window": SlidingWindowUCBPolicy(n_arms=3, window_size=40),
        },
        environment=BanditEnvironment(
            means=np.array([0.4, 0.6, 0.7]),
            reward_model=GaussianReward(std=0.05),
            dynamics=AbruptShift(frequency=120, std=0.4),
        ),
        config=ExperimentConfig(horizon=300, n_replicates=80, seed=41),
    ).summary()


def gradual_market_drift() -> list[dict[str, str | float | int]]:
    """Model slowly changing demand with gradual reward drift."""

    return compare(
        {
            "ucb": UCBPolicy(n_arms=3),
            "sliding-window": SlidingWindowUCBPolicy(n_arms=3, window_size=50),
        },
        environment=BanditEnvironment(
            means=np.array([0.5, 0.6, 0.7]),
            reward_model=GaussianReward(std=0.03),
            dynamics=GradualDrift(std=0.01),
        ),
        config=ExperimentConfig(horizon=300, n_replicates=80, seed=51),
    ).summary()


if __name__ == "__main__":
    scenarios = {
        "recommendation_clicks": recommendation_clicks(),
        "ad_allocation_recent_optimal_rate": ad_allocation_by_segment(),
        "pricing_experiment": pricing_experiment(),
        "clinical_trial_bernoulli": clinical_trial_bernoulli(),
        "proxy_server_selection": proxy_server_selection(),
        "non_stationary_demand": non_stationary_demand(),
        "gradual_market_drift": gradual_market_drift(),
    }
    for name, summary in scenarios.items():
        print(f"\n{name}")
        print(summary)
