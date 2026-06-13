import unittest

import numpy as np

from pymab.environments import LinearContextualEnvironment
from pymab.policies import (
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
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

    def test_contextual_experiment_runs(self) -> None:
        env = LinearContextualEnvironment(
            theta=np.array([[1.0, 0.0], [0.0, 1.0]]),
            context_provider=fixed_context,
        )
        result = Experiment(
            environment=env,
            policies=[
                LinUCBPolicy(n_arms=2, n_features=2),
                LinearThompsonSamplingPolicy(n_arms=2, n_features=2),
            ],
            config=ExperimentConfig(n_episodes=2, n_steps=5, seed=3),
        ).run()
        self.assertEqual(result.rewards.shape, (2, 5, 2))
        self.assertIsNone(result.q_values)


if __name__ == "__main__":
    unittest.main()
