import unittest
from typing import Tuple

import numpy as np
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
        self.n_bandits = 3
        self.policy = DummyPolicy(
            self.n_bandits,
            optimistic_initilization=1.0,
            variance=1.0,
            reward_distribution="gaussian",
        )

    def test_initialization(self):
        self.assertEqual(self.policy.n_bandits, self.n_bandits)
        self.assertEqual(self.policy.optimistic_initilization, 1.0)
        self.assertEqual(self.policy.variance, 1.0)
        self.assertTrue(
            issubclass(self.policy.reward_distribution, GaussianRewardDistribution)
        )
        self.assertTrue(
            np.array_equal(self.policy.times_selected, np.zeros(self.n_bandits))
        )
        self.assertTrue(
            np.array_equal(
                self.policy.actions_estimated_reward, np.full(self.n_bandits, 1.0)
            )
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
        self.assertTrue(np.array_equal(self.policy.Q_values, [0.1, 0.2, 0.3]))
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
        self.assertTrue(
            np.array_equal(self.policy.times_selected, np.zeros(self.n_bandits))
        )
        self.assertTrue(
            np.array_equal(
                self.policy.actions_estimated_reward, np.full(self.n_bandits, 1.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
