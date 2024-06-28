import unittest
from typing import Tuple
from unittest.mock import patch, MagicMock

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from pymab.reward_distribution import (
    GaussianRewardDistribution,
    BernoulliRewardDistribution,
    UniformRewardDistribution,
)
from pymab.policies.policy import Policy


class DummyPolicy(Policy):
    def select_action(self) -> Tuple[int, float]:
        action = np.argmax(self.actions_estimated_reward)
        reward = self._update(action)
        return action, reward


class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.policy_class = DummyPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 1.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.policy = self.policy_class(
            self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
        )

    def test_optimistic_initialization(self):
        assert_array_equal(
            self.policy.actions_estimated_reward,
            np.full(self.n_bandits, self.optimistic_initialization),
        )

    def test_non_optimistic_initialization(self):
        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=0.0,
            variance=1.0,
            reward_distribution="gaussian",
        )
        assert_array_equal(
            self.policy.actions_estimated_reward, np.zeros(self.n_bandits)
        )

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0
        for i in range(self.n_bandits):
            max_estimated_action = np.argmax(self.policy.actions_estimated_reward)
            action, reward = self.policy.select_action()
            total_reward += reward

            self.assertEqual(action, max_estimated_action)
            self.assertTrue(self.policy.times_selected[action] > 0)
            self.assertTrue(self.policy.total_reward == total_reward)

    def test_update_mock_rewards(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0

        mock_action = 0
        mock_reward = self.policy.Q_values[mock_action]
        with patch.object(
            self.policy, "_get_actual_reward", return_value=mock_reward
        ) as mock_method:
            reward = self.policy._update(mock_action)
            total_reward += reward

            assert_array_equal(self.policy.times_selected, np.array([1, 0, 0]))
            self.assertEqual(self.policy.total_reward, mock_reward)

            assert_allclose(
                self.policy.actions_estimated_reward,
                np.array(
                    [
                        mock_reward,
                        self.optimistic_initialization,
                        self.optimistic_initialization,
                    ]
                ),
                rtol=1e-5,
                atol=1e-8,
            )

            mock_method.assert_called_once_with(mock_action)

        mock_action = 1
        mock_reward = self.policy.Q_values[mock_action]
        with patch.object(
            self.policy, "_get_actual_reward", return_value=mock_reward
        ) as mock_method:
            reward = self.policy._update(mock_action)
            total_reward += reward

            assert_array_equal(self.policy.times_selected, np.array([1, 1, 0]))
            self.assertTrue(self.policy.total_reward == 0.6)

            assert_allclose(
                self.policy.actions_estimated_reward,
                np.array(
                    [
                        0.1,
                        mock_reward,
                        self.optimistic_initialization,
                    ]
                ),
                rtol=1e-5,
                atol=1e-8,
            )

            mock_method.assert_called_once_with(mock_action)

        mock_action = 2
        mock_reward = self.policy.Q_values[mock_action]
        with patch.object(
            self.policy, "_get_actual_reward", return_value=mock_reward
        ) as mock_method:
            reward = self.policy._update(mock_action)
            total_reward += reward

            assert_array_equal(self.policy.times_selected, np.array([1, 1, 1]))
            self.assertTrue(self.policy.total_reward == 1.5)

            assert_allclose(
                self.policy.actions_estimated_reward,
                np.array(
                    [
                        0.1,
                        0.5,
                        mock_reward,
                    ]
                ),
                rtol=1e-5,
                atol=1e-8,
            )

            mock_method.assert_called_once_with(mock_action)

        mock_action = 2
        mock_reward = 1.2
        with patch.object(
            self.policy, "_get_actual_reward", return_value=mock_reward
        ) as mock_method:
            action, reward = self.policy.select_action()
            total_reward += reward

            self.assertEqual(action, 2)
            assert_array_equal(self.policy.times_selected, np.array([1, 1, 2]))
            self.assertTrue(self.policy.total_reward == 2.7)

            assert_allclose(
                self.policy.actions_estimated_reward,
                np.array(
                    [
                        0.1,
                        0.5,
                        (0.9 + mock_reward) / self.policy.times_selected[mock_action],
                    ]
                ),
                rtol=1e-5,
                atol=1e-8,
            )

            mock_method.assert_called_once_with(mock_action)

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

    def test_get_reward_distribution(self):
        self.assertTrue(
            issubclass(
                self.policy.get_reward_distribution("gaussian"),
                GaussianRewardDistribution,
            )
        )
        self.assertTrue(
            issubclass(
                self.policy.get_reward_distribution("bernoulli"),
                BernoulliRewardDistribution,
            )
        )
        self.assertTrue(
            issubclass(
                self.policy.get_reward_distribution("uniform"),
                UniformRewardDistribution,
            )
        )
        with self.assertRaises(ValueError):
            self.policy.get_reward_distribution("invalid")

    def test_get_actual_reward(self):
        self.policy._Q_values = np.array([0.5, 0.6, 0.7])
        reward = self.policy._get_actual_reward(0)
        self.assertTrue(-2.5 <= reward <= 3.5)  # 3 standard deviations range

    def test_update(self):
        self.policy._Q_values = np.array([0.5, 0.6, 0.7])
        chosen_action = 0
        initial_reward = self.policy.actions_estimated_reward[chosen_action]
        self.policy._update(chosen_action)
        self.assertEqual(self.policy.times_selected[chosen_action], 1)
        self.assertNotEqual(
            self.policy.actions_estimated_reward[chosen_action], initial_reward
        )

    def test_Q_values_property(self):
        self.policy.Q_values = [0.1, 0.2, 0.3]
        assert_array_equal(self.policy.Q_values, [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            self.policy.Q_values = [0.1, 0.2]

    def test_reset(self):
        self.policy.current_step = 5
        self.policy.total_reward = 10.0
        self.policy.times_selected = np.array([1, 1, 1])
        self.policy.actions_estimated_reward = np.array([0.5, 0.5, 0.5])
        self.policy.reset()
        self.assertEqual(self.policy.current_step, 0)
        self.assertEqual(self.policy.total_reward, 0)
        assert_array_equal(self.policy.times_selected, np.zeros(self.n_bandits))
        assert_array_equal(
            self.policy.actions_estimated_reward,
            np.full(self.n_bandits, self.policy.optimistic_initialization),
        )


if __name__ == "__main__":
    unittest.main()
