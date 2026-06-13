import unittest

import numpy as np

from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import BanditEnvironment, LinearContextualEnvironment
from pymab.policies import (
    BernoulliThompsonSamplingPolicy,
    GreedyPolicy,
    LinUCBPolicy,
    UCBPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig


def _normal_lower_confidence_bound(
    success_rate: float, n_observations: int, z_score: float = 1.96
) -> float:
    variance = success_rate * (1.0 - success_rate) / n_observations
    return success_rate - z_score * float(np.sqrt(variance))


class StatisticalConfidenceTests(unittest.TestCase):
    def test_ucb_learns_stationary_gaussian_best_arm_with_confidence(self) -> None:
        result = Experiment(
            environment=BanditEnvironment(
                q_values=np.array([0.0, 0.25, 1.0]),
                reward_distribution=GaussianReward(std=0.05),
            ),
            policies=[UCBPolicy(n_arms=3), GreedyPolicy(n_arms=3)],
            config=ExperimentConfig(n_episodes=80, n_steps=80, seed=2026),
        ).run()

        ucb_recent_rate = float(result.optimal_action_rate_by_step[-20:, 0].mean())
        greedy_recent_rate = float(result.optimal_action_rate_by_step[-20:, 1].mean())
        lower_bound = _normal_lower_confidence_bound(
            ucb_recent_rate, n_observations=80 * 20
        )

        self.assertGreater(lower_bound, 0.90)
        self.assertGreater(ucb_recent_rate, greedy_recent_rate + 0.25)
        self.assertLess(
            result.cumulative_regret[-1, 0],
            result.cumulative_regret[-1, 1] * 0.5,
        )

    def test_bernoulli_thompson_sampling_finds_best_arm_with_confidence(self) -> None:
        result = Experiment(
            environment=BanditEnvironment(
                q_values=np.array([0.1, 0.35, 0.8]),
                reward_distribution=BernoulliReward(),
            ),
            policies=[BernoulliThompsonSamplingPolicy(n_arms=3)],
            config=ExperimentConfig(n_episodes=120, n_steps=100, seed=99),
        ).run()

        recent_rate = float(result.optimal_action_rate_by_step[-20:, 0].mean())
        lower_bound = _normal_lower_confidence_bound(
            recent_rate, n_observations=120 * 20
        )

        self.assertGreater(lower_bound, 0.94)
        self.assertLess(result.cumulative_regret[-1, 0], 8.0)

    def test_linucb_learns_context_dependent_best_arm_with_confidence(self) -> None:
        def context_provider(rng: np.random.Generator) -> np.ndarray:
            if rng.random() < 0.5:
                return np.array([[1.0, 0.0], [1.0, 0.0]])
            return np.array([[0.0, 1.0], [0.0, 1.0]])

        result = Experiment(
            environment=LinearContextualEnvironment(
                theta=np.array([[1.0, 0.0], [0.0, 1.0]]),
                context_provider=context_provider,
                reward_distribution=GaussianReward(std=0.02),
            ),
            policies=[LinUCBPolicy(n_arms=2, n_features=2)],
            config=ExperimentConfig(n_episodes=100, n_steps=80, seed=5),
        ).run()

        recent_rate = float(result.optimal_action_rate_by_step[-20:, 0].mean())
        lower_bound = _normal_lower_confidence_bound(
            recent_rate, n_observations=100 * 20
        )

        self.assertGreater(lower_bound, 0.98)
        self.assertLess(result.cumulative_regret[-1, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
