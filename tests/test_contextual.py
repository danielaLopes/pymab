import unittest

import numpy as np

from pymab.distributions import BernoulliReward
from pymab.environments import LinearContextualEnvironment
from pymab.policies import (
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig


def fixed_context(rng: np.random.Generator) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


class ContextualPolicyTests(unittest.TestCase):
    def test_linear_epsilon_greedy_updates_only_action_row(self) -> None:
        policy = LinearEpsilonGreedyPolicy(
            n_arms=2, n_features=2, epsilon=0.0, learning_rate=0.5
        )
        context = fixed_context(np.random.default_rng(1))
        policy.update(action=1, reward=2.0, context=context)
        np.testing.assert_allclose(policy.theta[0], np.zeros(2))
        self.assertGreater(policy.theta[1, 1], 0)

    def test_linucb_scores_have_one_value_per_arm(self) -> None:
        policy = LinUCBPolicy(n_arms=2, n_features=2, alpha=1.0)
        scores = policy.upper_confidence_bounds(fixed_context(np.random.default_rng(1)))
        self.assertEqual(scores.shape, (2,))

    def test_logistic_contextual_policy_updates_selected_arm(self) -> None:
        policy = LogisticContextualBanditPolicy(
            n_arms=2, n_features=2, epsilon=0.0, learning_rate=0.5
        )
        context = fixed_context(np.random.default_rng(1))
        before = policy.predicted_probabilities(context)
        policy.update(action=0, reward=1.0, context=context)
        after = policy.predicted_probabilities(context)
        self.assertGreater(after[0], before[0])
        np.testing.assert_allclose(policy.theta[1], np.zeros(2))

    def test_contextual_experiment_runs(self) -> None:
        env = LinearContextualEnvironment(
            theta=np.array([[1.0, 0.0], [0.0, 1.0]]),
            context_provider=fixed_context,
            reward_distribution=BernoulliReward(),
        )
        result = Experiment(
            environment=env,
            policies=[
                LinUCBPolicy(n_arms=2, n_features=2),
                LinearThompsonSamplingPolicy(n_arms=2, n_features=2),
                LogisticContextualBanditPolicy(n_arms=2, n_features=2),
            ],
            config=ExperimentConfig(n_episodes=2, n_steps=5, seed=3),
        ).run()
        self.assertEqual(result.rewards.shape, (2, 5, 3))
        self.assertIsNone(result.q_values)


if __name__ == "__main__":
    unittest.main()
