import math
import unittest
from unittest.mock import patch

import numpy as np
from pymab.policies.ucb import StationaryUCBPolicy
from tests.policies.test_policy import TestPolicy


class TestUCBPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = StationaryUCBPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 1.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.c = 1
        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
            c=self.c,
        )

    def test_initialization(self):
        super().test_initialization()
        self.assertEqual(self.policy.c, self.c)

    def test_select_action(self):
        self.policy.Q_values = [0.1, 0.5, 0.9]
        total_reward = 0

        for i in range(0, self.n_bandits):
            action, reward = self.policy.select_action()
            total_reward += reward

            self.assertEqual(action, i)
            self.assertTrue(self.policy.times_selected[i] == 1)
            self.assertEqual(self.policy.total_reward, total_reward)

        ucb_values = np.zeros(self.n_bandits)
        for action_index in range(0, self.n_bandits):
            if self.policy.times_selected[action_index] > 0:
                ucb_values[action_index] = math.sqrt(
                    self.c
                    * math.log(self.policy.current_step + 1)
                    / self.policy.times_selected[action_index]
                )
        max_estimated_action = np.argmax(
            self.policy.actions_estimated_reward + ucb_values
        )
        action, reward = self.policy.select_action()
        total_reward += reward

        self.assertEqual(action, max_estimated_action)
        self.assertTrue(self.policy.times_selected[action] > 0)
        self.assertTrue(self.policy.total_reward == total_reward)

    def test_ucb_values(self):
        self.policy.times_selected = np.array([10, 5, 2])
        self.policy.actions_estimated_reward = np.array([0.1, 0.2, 0.3])
        self.policy.current_step = 18

        expected_ucb_values = np.array(
            [
                math.sqrt(
                    self.policy.c
                    * math.log(self.policy.current_step + 1)
                    / self.policy.times_selected[0]
                ),
                math.sqrt(
                    self.policy.c
                    * math.log(self.policy.current_step + 1)
                    / self.policy.times_selected[1]
                ),
                math.sqrt(
                    self.policy.c
                    * math.log(self.policy.current_step + 1)
                    / self.policy.times_selected[2]
                ),
            ]
        )
        with patch.object(self.policy, "_update", return_value=0.9):
            action, reward = self.policy.select_action()
            calculated_ucb_values = (
                self.policy.actions_estimated_reward + expected_ucb_values
            )
            self.assertEqual(action, np.argmax(calculated_ucb_values))


if __name__ == "__main__":
    unittest.main()
