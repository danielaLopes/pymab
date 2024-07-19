import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from pymab.policies.contextual_bandits import ContextualBanditPolicy
from tests.policies.test_policy import TestPolicy


class TestContextualBanditPolicy(TestPolicy):
    @staticmethod
    def context_func() -> np.ndarray:
        return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).T

    def setUp(self):
        self.policy_class = ContextualBanditPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 1.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.context_dim = 2
        self.learning_rate = 0.1

        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
            context_dim=self.context_dim,
            context_func=self.context_func,
            learning_rate=self.learning_rate,
        )

    def test_non_optimistic_initialization(self):
        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=0.0,
            variance=1.0,
            reward_distribution=self.reward_distribution,
            context_dim=self.context_dim,
            context_func=self.context_func,
            learning_rate=self.learning_rate,
        )
        assert_array_equal(
            self.policy.actions_estimated_reward, np.zeros(self.n_bandits)
        )

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0
        context = self.context_func()
        for i in range(self.n_bandits):
            expected_rewards = np.array(
                [self.policy.theta[i] @ context[:, i] for i in range(self.n_bandits)]
            )
            max_estimated_action = np.argmax(expected_rewards)
            action, reward = self.policy.select_action(context=context)
            total_reward += reward

            self.assertEqual(action, max_estimated_action)
            self.assertTrue(self.policy.times_selected[action] > 0)
            self.assertTrue(self.policy.total_reward == total_reward)

    def test_select_action_wrong_context_shapes(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        context = np.array([[0], [0], [0]])

        with self.assertRaises(ValueError) as cm:
            self.policy.select_action(context=context)
        self.assertEqual(str(cm.exception), "Context dimension does not match the expected context_dim.")

        context = np.array([[0, 0], [0, 0]])
        with self.assertRaises(ValueError) as cm:
            self.policy.select_action(context=context)
        self.assertEqual(str(cm.exception), "Context dimension does not match the expected n_bandits.")

    def test_update(self):
        self.policy._Q_values = np.array([0.5, 0.6, 0.7])
        chosen_action = 0
        initial_reward = self.policy.actions_estimated_reward[chosen_action]
        self.policy._update(chosen_action, context_chosen_action=self.context_func()[:, chosen_action])
        self.assertEqual(self.policy.times_selected[chosen_action], 1)
        self.assertNotEqual(
            self.policy.actions_estimated_reward[chosen_action], initial_reward
        )

    def test_update_mock_rewards(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0

        mock_action = 0
        mock_reward = self.policy.Q_values[mock_action]
        with patch.object(
            self.policy, "_get_actual_reward", return_value=mock_reward
        ) as mock_method:
            reward = self.policy._update(mock_action, context_chosen_action=self.context_func()[:, mock_action])
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
            reward = self.policy._update(mock_action, context_chosen_action=self.context_func()[:, mock_action])
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
            reward = self.policy._update(mock_action, context_chosen_action=self.context_func()[:, mock_action])
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
            reward = self.policy._update(mock_action, context_chosen_action=self.context_func()[:, mock_action])
            total_reward += reward

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


if __name__ == "__main__":
    unittest.main()
