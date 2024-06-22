import unittest
import numpy as np
from pymab.policies.greedy import GreedyPolicy
from pymab.reward_distribution import GaussianRewardDistribution, BernoulliRewardDistribution, UniformRewardDistribution
from tests.policies.test_policy import TestPolicy


class TestGreedyPolicy(TestPolicy):

    def setUp(self):
        self.n_bandits = 3
        self.policy = GreedyPolicy(
            n_bandits=self.n_bandits,
            optimistic_initilization=1.0,
            variance=1.0,
            reward_distribution="gaussian"
        )

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        action, reward = self.policy.select_action()
        self.assertEqual(action, np.argmax(self.policy.actions_estimated_reward))
        self.assertTrue(self.policy.times_selected[action] > 0)
        self.assertTrue(self.policy.total_reward > 0)


if __name__ == '__main__':
    unittest.main()