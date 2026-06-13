import unittest

import numpy as np

from pymab.policies import (
    BayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    EpsilonGreedyPolicy,
    GaussianThompsonSamplingPolicy,
    GradientBanditPolicy,
    GreedyPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)


class PolicyTests(unittest.TestCase):
    def test_greedy_updates_incremental_average(self) -> None:
        policy = GreedyPolicy(n_arms=2)
        policy.update(action=1, reward=1.0)
        policy.update(action=1, reward=3.0)
        self.assertEqual(policy.counts[1], 2)
        self.assertEqual(policy.estimates[1], 2.0)

    def test_epsilon_validation(self) -> None:
        with self.assertRaises(ValueError):
            EpsilonGreedyPolicy(n_arms=2, epsilon=1.1)

    def test_ucb_selects_each_arm_once(self) -> None:
        policy = UCBPolicy(n_arms=3)
        rng = np.random.default_rng(1)
        self.assertEqual(policy.select_action(rng=rng), 0)
        policy.update(action=0, reward=0.0)
        self.assertEqual(policy.select_action(rng=rng), 1)
        policy.update(action=1, reward=0.0)
        self.assertEqual(policy.select_action(rng=rng), 2)

    def test_sliding_window_forgets_old_rewards(self) -> None:
        policy = SlidingWindowUCBPolicy(n_arms=1, window_size=2)
        policy.update(action=0, reward=1.0)
        policy.update(action=0, reward=3.0)
        policy.update(action=0, reward=5.0)
        self.assertEqual(policy.estimates[0], 4.0)

    def test_discounted_ucb_uses_discounted_counts(self) -> None:
        policy = DiscountedUCBPolicy(n_arms=1, discount_factor=0.5)
        policy.update(action=0, reward=1.0)
        policy.update(action=0, reward=1.0)
        self.assertLess(policy.discounted_counts[0], policy.counts[0])

    def test_softmax_probabilities_sum_to_one(self) -> None:
        policy = SoftmaxPolicy(n_arms=3, temperature=0.5)
        np.testing.assert_allclose(np.sum(policy.action_probabilities()), 1.0)

    def test_gradient_preferences_update(self) -> None:
        policy = GradientBanditPolicy(n_arms=2, learning_rate=0.1)
        action = policy.select_action(rng=np.random.default_rng(1))
        before = policy.preferences.copy()
        policy.update(action=action, reward=1.0)
        self.assertFalse(np.allclose(before, policy.preferences))

    def test_bernoulli_thompson_requires_binary_rewards(self) -> None:
        policy = BernoulliThompsonSamplingPolicy(n_arms=2)
        with self.assertRaises(ValueError):
            policy.update(action=0, reward=0.2)

    def test_thompson_factory_preserves_distribution(self) -> None:
        bernoulli = ThompsonSamplingPolicy(n_arms=2, reward_distribution="bernoulli")
        gaussian = ThompsonSamplingPolicy(n_arms=2, reward_distribution="gaussian")
        self.assertIsInstance(bernoulli, BernoulliThompsonSamplingPolicy)
        self.assertIsInstance(gaussian, GaussianThompsonSamplingPolicy)

    def test_bayesian_ucb_factory(self) -> None:
        policy = BayesianUCBPolicy(n_arms=2, reward_distribution="gaussian")
        action = policy.select_action(rng=np.random.default_rng(1))
        self.assertIn(action, {0, 1})


if __name__ == "__main__":
    unittest.main()
