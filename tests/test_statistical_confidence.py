import unittest

import numpy as np

from pymab.benchmarking import bootstrap_mean_interval
from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import (
    BanditEnvironment,
    EnvironmentDynamics,
    LinearContextualEnvironment,
)
from pymab.policies import (
    BernoulliThompsonSamplingPolicy,
    GreedyPolicy,
    LinUCBPolicy,
    RandomPolicy,
    SlidingWindowUCBPolicy,
    UCBPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig
from pymab.types import RewardDomain


def _replicate_mean_lower_bound(values: np.ndarray, *, seed: int) -> float:
    lower, _ = bootstrap_mean_interval(values, n_resamples=2_000, seed=seed)
    if lower is None:
        raise AssertionError("at least two independent replicates are required")
    return lower


class StatisticalConfidenceTests(unittest.TestCase):
    def test_ucb_outperforms_random_on_stationary_gaussian(self) -> None:
        result = Experiment(
            environment=BanditEnvironment(
                means=np.array([0.0, 0.25, 1.0]),
                reward_model=GaussianReward(std=0.05),
            ),
            policies={
                "ucb": UCBPolicy(n_arms=3),
                "random": RandomPolicy(n_arms=3),
                "greedy": GreedyPolicy(n_arms=3),
            },
            config=ExperimentConfig(n_replicates=80, horizon=80, seed=2026),
        ).run()

        replicate_rates = np.mean(result.optimal_action_indicator[:, -20:, :], axis=1)
        ucb_recent_rate = float(np.mean(replicate_rates[:, 0]))
        random_recent_rate = float(np.mean(replicate_rates[:, 1]))
        greedy_recent_rate = float(np.mean(replicate_rates[:, 2]))
        lower_bound = _replicate_mean_lower_bound(replicate_rates[:, 0], seed=1)

        self.assertGreater(lower_bound, 0.90)
        self.assertGreater(ucb_recent_rate, random_recent_rate + 0.45)
        self.assertGreater(ucb_recent_rate, greedy_recent_rate + 0.25)
        self.assertLess(
            result.cumulative_regret[-1, 0],
            result.cumulative_regret[-1, 1] * 0.5,
        )

    def test_bernoulli_thompson_sampling_finds_best_arm_with_confidence(self) -> None:
        result = Experiment(
            environment=BanditEnvironment(
                means=np.array([0.1, 0.35, 0.8]),
                reward_model=BernoulliReward(),
            ),
            policies={
                "thompson": BernoulliThompsonSamplingPolicy(n_arms=3),
                "random": RandomPolicy(n_arms=3),
            },
            config=ExperimentConfig(n_replicates=120, horizon=100, seed=99),
        ).run()

        replicate_rates = np.mean(result.optimal_action_indicator[:, -20:, :], axis=1)
        recent_rate = float(np.mean(replicate_rates[:, 0]))
        random_recent_rate = float(np.mean(replicate_rates[:, 1]))
        lower_bound = _replicate_mean_lower_bound(replicate_rates[:, 0], seed=2)

        self.assertGreater(lower_bound, 0.94)
        self.assertGreater(recent_rate, random_recent_rate + 0.50)
        self.assertLess(result.cumulative_regret[-1, 0], 8.0)
        self.assertLess(
            result.cumulative_regret[-1, 0],
            result.cumulative_regret[-1, 1] * 0.35,
        )

    def test_linucb_learns_context_dependent_best_arm_with_confidence(self) -> None:
        def context_provider(rng: np.random.Generator) -> np.ndarray:
            if rng.random() < 0.5:
                return np.array([[1.0, 0.0], [1.0, 0.0]])
            return np.array([[0.0, 1.0], [0.0, 1.0]])

        result = Experiment(
            environment=LinearContextualEnvironment(
                theta=np.array([[1.0, 0.0], [0.0, 1.0]]),
                context_provider=context_provider,
                reward_model=GaussianReward(std=0.02),
            ),
            policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
            config=ExperimentConfig(n_replicates=100, horizon=80, seed=5),
        ).run()

        replicate_rates = np.mean(result.optimal_action_indicator[:, -20:, 0], axis=1)
        lower_bound = _replicate_mean_lower_bound(replicate_rates, seed=3)

        self.assertGreater(lower_bound, 0.98)
        self.assertLess(result.cumulative_regret[-1, 0], 2.0)

    def test_sliding_window_ucb_recovers_after_abrupt_shift(self) -> None:
        class FlipBestArm(EnvironmentDynamics):
            supported_domains = frozenset({RewardDomain.REAL})

            def apply(
                self,
                q_values: np.ndarray,
                *,
                step: int,
                rng: np.random.Generator,
            ) -> np.ndarray:
                if step < 40:
                    return np.array([1.0, 0.0])
                return np.array([0.0, 1.0])

        result = Experiment(
            environment=BanditEnvironment(
                means=np.array([1.0, 0.0]),
                reward_model=GaussianReward(std=0.01),
                dynamics=FlipBestArm(),
            ),
            policies={
                "sliding-window": SlidingWindowUCBPolicy(
                    n_arms=2, c=1.0, window_size=10
                ),
                "ucb": UCBPolicy(n_arms=2, c=1.0),
            },
            config=ExperimentConfig(n_replicates=80, horizon=100, seed=42),
        ).run()

        early_post_shift_rates = np.mean(
            result.optimal_action_indicator[:, 40:60, :], axis=1
        )
        paired_difference = early_post_shift_rates[:, 0] - early_post_shift_rates[:, 1]
        lower, _ = bootstrap_mean_interval(paired_difference, n_resamples=2_000, seed=4)
        self.assertIsNotNone(lower)
        self.assertGreater(float(lower), 0.05)


if __name__ == "__main__":
    unittest.main()
