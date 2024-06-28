import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from pymab.policies.epsilon_greedy import EpsilonGreedyPolicy
from tests.policies.test_policy import TestPolicy


class TestEpsilonGreedyPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = EpsilonGreedyPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 1.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.epsilon = 0.1
        self.policy = self.policy_class(
            self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
            epsilon=self.epsilon,
        )

    def test_initialization(self):
        super().test_initialization()
        self.assertEqual(self.policy.epsilon, self.epsilon)

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        exploit_value = 0.9
        explore_value = 0.0

        total_reward = 0

        with patch("random.uniform", return_value=explore_value) as mock_method:
            exploit_action = np.argmax(self.policy.actions_estimated_reward)
            action, reward = self.policy.select_action()
            total_reward += reward

            self.assertNotEqual(action, exploit_action)
            self.assertTrue(self.policy.times_selected[action] > 0)
            self.assertTrue(self.policy.total_reward == total_reward)

            mock_method.assert_called_once()

        with patch("random.uniform", return_value=exploit_value) as mock_method:
            exploit_action = np.argmax(self.policy.actions_estimated_reward)
            action, reward = self.policy.select_action()
            total_reward += reward

            self.assertEqual(action, exploit_action)
            self.assertTrue(self.policy.times_selected[action] > 0)
            self.assertTrue(self.policy.total_reward == total_reward)

            mock_method.assert_called_once()


if __name__ == "__main__":
    unittest.main()
