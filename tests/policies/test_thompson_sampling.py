import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from pymab.policies.policy import Policy
from pymab.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    ThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
)
from pymab.reward_distribution import (
    BernoulliRewardDistribution,
    GaussianRewardDistribution,
)
from tests.policies.test_policy import TestPolicy


class TestBernoulliThompsonSamplingPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = BernoulliThompsonSamplingPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 0.0
        self.variance = 1.0
        self.reward_distribution = "bernoulli"
        self.policy = ThompsonSamplingPolicy(
            self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
        )

    def test_initialization(self):
        self.assertEqual(self.policy.n_bandits, self.n_bandits)
        self.assertEqual(
            self.policy.optimistic_initialization, self.optimistic_initialization
        )
        self.assertEqual(self.policy.variance, self.variance)
        self.assertTrue(
            issubclass(self.policy.reward_distribution, BernoulliRewardDistribution)
        )
        assert_array_equal(self.policy.times_selected, np.zeros(self.n_bandits))
        assert_array_equal(
            self.policy.actions_estimated_reward,
            np.full(self.n_bandits, self.optimistic_initialization),
        )

        self.assertTrue(isinstance(self.policy, self.policy_class))
        assert_array_equal(self.policy.times_success, np.zeros(self.n_bandits))
        assert_array_equal(self.policy.times_failure, np.zeros(self.n_bandits))

    def test_update(self):
        self.policy._Q_values = np.array([0.5, 0.6, 0.7])

        success_value = 1
        failure_value = 0

        action = 0
        success_before = self.policy.times_success[action]
        with patch.object(Policy, "_update", return_value=success_value) as mock_method:
            reward = self.policy._update(action)
            self.assertEqual(success_before + 1, self.policy.times_success[action])
            self.assertEqual(reward, 1)

            mock_method.assert_called_once_with(action)

        failure_before = self.policy.times_failure[action]
        with patch.object(Policy, "_update", return_value=failure_value) as mock_method:
            reward = self.policy._update(action)
            self.assertEqual(failure_before + 1, self.policy.times_failure[action])
            self.assertEqual(reward, 0)

            mock_method.assert_called_once_with(action)

    @patch("numpy.random.beta")
    def test_select_action(self, mock_beta):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0

        mock_beta.side_effect = [
            0.1,
            0.2,
            0.3,  # First call to np.random.beta
            0.9,
            0.8,
            0.7,  # Second call to np.random.beta
            0.1,
            0.5,
            0.2,  # Third call to np.random.beta
        ]

        action, reward = self.policy.select_action()
        total_reward += reward
        self.assertEqual(action, 2)
        self.assertEqual(self.policy.times_selected[action], 1)
        self.assertEqual(self.policy.total_reward, total_reward)

        action, reward = self.policy.select_action()
        total_reward += reward
        self.assertEqual(action, 0)
        self.assertEqual(self.policy.times_selected[action], 1)
        self.assertEqual(self.policy.total_reward, total_reward)

        action, reward = self.policy.select_action()
        total_reward += reward
        self.assertEqual(action, 1)
        self.assertEqual(self.policy.times_selected[action], 1)
        self.assertEqual(self.policy.total_reward, total_reward)


class TestGaussianThompsonSamplingPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = GaussianThompsonSamplingPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 0.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.policy = ThompsonSamplingPolicy(
            self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
        )

    def test_initialization(self):
        self.assertEqual(self.policy.n_bandits, self.n_bandits)
        self.assertEqual(
            self.policy.optimistic_initialization, self.optimistic_initialization
        )
        self.assertEqual(self.policy.variance, self.variance)
        self.assertTrue(
            issubclass(self.policy.reward_distribution, GaussianRewardDistribution)
        )
        assert_array_equal(self.policy.times_selected, np.zeros(self.n_bandits))
        assert_array_equal(
            self.policy.actions_estimated_reward,
            np.full(self.n_bandits, self.optimistic_initialization),
        )

        self.assertTrue(isinstance(self.policy, self.policy_class))
        assert_array_equal(self.policy.means, np.zeros(self.n_bandits))
        assert_array_equal(
            self.policy.precisions, np.ones(self.n_bandits) / self.variance
        )

    def test_update(self):
        self.policy._Q_values = np.array([0.5, 0.6, 0.7])

        mock_value = 1

        action = 0
        prior_mean_before = self.policy.means[action]
        prior_precision_before = self.policy.precisions[action]
        with patch.object(Policy, "_update", return_value=mock_value) as mock_method:
            reward = self.policy._update(action)
            prior_mean_expected = (
                prior_precision_before * prior_mean_before + reward
            ) / (prior_precision_before + 1)
            prior_precision_expected = prior_precision_before + 1
            self.assertEqual(prior_mean_expected, self.policy.means[action])
            self.assertEqual(prior_precision_expected, self.policy.precisions[action])

            mock_method.assert_called_once_with(action)

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0

        with patch("numpy.random.normal", side_effect=[0.1, 0.2, 0.3, 0.4]) as mock_normal:
            action, reward = self.policy.select_action()
            total_reward += reward
            self.assertEqual(action, 2)
            self.assertEqual(self.policy.times_selected[action], 1)
            self.assertEqual(self.policy.total_reward, total_reward)

            self.assertEqual(mock_normal.call_count, 4)

        with patch("numpy.random.normal", side_effect=[0.3, 0.2, 0.2, 0.4]) as mock_normal:
            action, reward = self.policy.select_action()
            total_reward += reward
            self.assertEqual(action, 0)
            self.assertEqual(self.policy.times_selected[action], 1)
            self.assertEqual(self.policy.total_reward, total_reward)

            self.assertEqual(mock_normal.call_count, 4)

        with patch("numpy.random.normal", side_effect=[0.1, 0.3, 0.2, 0.4]) as mock_normal:
            action, reward = self.policy.select_action()
            total_reward += reward
            self.assertEqual(action, 1)
            self.assertEqual(self.policy.times_selected[action], 1)
            self.assertEqual(self.policy.total_reward, total_reward)

            self.assertEqual(mock_normal.call_count, 4)


if __name__ == "__main__":
    unittest.main()
