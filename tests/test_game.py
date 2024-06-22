import unittest
import numpy as np

from pymab.policies.greedy import GreedyPolicy
from pymab.policies.policy import Policy
from pymab.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
)
from pymab.game import Game  # Import your Game class


class TestGame(unittest.TestCase):

    def setUp(self):
        self.n_episodes = 10
        self.n_steps = 100
        self.n_bandits = 3
        self.Q_values = [-0.3, 0.7, 0.1]
        self.Q_values_variance = 0.5
        self.reward_distribution = "gaussian"
        self.policies = [
            GreedyPolicy(
                optimistic_initilization=1,
                n_bandits=self.n_bandits,
                reward_distribution=self.reward_distribution,
            ),
            GaussianThompsonSamplingPolicy(
                self.n_bandits, reward_distribution=self.reward_distribution
            ),
        ]
        self.game = Game(
            n_episodes=self.n_episodes,
            n_steps=self.n_steps,
            policies=self.policies,
            n_bandits=self.n_bandits,
            Q_values=self.Q_values,
            Q_values_variance=self.Q_values_variance,
            is_stationary=False,
        )

    def test_initialization(self):
        self.assertEqual(self.game.n_episodes, self.n_episodes)
        self.assertEqual(self.game.n_steps, self.n_steps)
        self.assertEqual(self.game.n_bandits, self.n_bandits)
        self.assertEqual(len(self.game.policies), len(self.policies))

    def test_initialization_multiple_reward_distributions(self):
        self.policies.append(
            BernoulliThompsonSamplingPolicy(
                self.n_bandits, reward_distribution="bernoulli"
            )
        )
        with self.assertRaises(ValueError):
            self.game = Game(
                n_episodes=self.n_episodes,
                n_steps=self.n_steps,
                policies=self.policies,
                n_bandits=self.n_bandits,
                Q_values=self.Q_values,
                Q_values_variance=self.Q_values_variance,
                is_stationary=False,
            )

    def test_generate_Q_values(self):
        for i in range(self.n_steps):
            self.game.generate_Q_values()
            self.assertEqual(len(self.game.Q_values), self.n_bandits)
            for Q_value in self.game.Q_values:
                self.assertTrue(
                    (Q_value - 3 * self.Q_values_variance)
                    <= Q_value
                    <= (Q_value + 3 * self.Q_values_variance),
                    "Generated Q_value is not within the expected range.",
                )

    def test_new_episode(self):
        self.game.new_episode(0)
        for policy in self.game.policies:
            self.assertTrue(np.array_equal(policy.Q_values, self.game.Q_values))

    def test_game_loop(self):
        self.game.game_loop()
        self.assertEqual(
            self.game.rewards_by_policy.shape,
            (self.n_episodes, self.n_steps, len(self.policies)),
        )
        self.assertEqual(
            self.game.actions_selected_by_policy.shape,
            (self.n_episodes, self.n_steps, len(self.policies)),
        )

    def test_average_rewards_by_step(self):
        self.game.game_loop()
        avg_rewards = self.game.average_rewards_by_step
        self.assertEqual(avg_rewards.shape, (self.n_steps, len(self.policies)))

    def test_average_rewards_by_episode(self):
        self.game.game_loop()
        avg_rewards = self.game.average_rewards_by_episode
        self.assertEqual(avg_rewards.shape, (self.n_episodes, len(self.policies)))

    def test_total_rewards_by_step(self):
        self.game.game_loop()
        total_rewards = self.game.total_rewards_by_step
        self.assertEqual(total_rewards.shape, (self.n_steps, len(self.policies)))

    def test_cumulative_regret_by_step(self):
        self.game.game_loop()
        cumulative_regret = self.game.cumulative_regret_by_step
        self.assertEqual(cumulative_regret.shape, (self.n_steps, len(self.policies)))


if __name__ == "__main__":
    unittest.main()
