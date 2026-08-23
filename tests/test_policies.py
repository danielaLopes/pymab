import unittest

import numpy as np
import pytest

from pymab.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    CUSUMUCBPolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    EpsilonGreedyPolicy,
    EXP3Policy,
    GaussianBayesianUCBPolicy,
    GradientBanditPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    MedianEliminationPolicy,
    MOSSPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    SuccessiveEliminationPolicy,
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

    def test_decaying_epsilon_decreases_to_floor(self) -> None:
        policy = DecayingEpsilonGreedyPolicy(
            n_arms=2, initial_epsilon=1.0, min_epsilon=0.1, decay_rate=1.0
        )
        start = policy.epsilon
        for _ in range(10):
            policy.update(action=0, reward=1.0)
        self.assertLess(policy.epsilon, start)
        self.assertGreaterEqual(policy.epsilon, 0.1)

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

    def test_kl_ucb_requires_binary_rewards_and_returns_indices(self) -> None:
        policy = KLUCBPolicy(n_arms=2)
        with self.assertRaises(ValueError):
            policy.update(action=0, reward=0.5)
        policy.update(action=0, reward=1.0)
        policy.update(action=1, reward=0.0)
        indices = policy.indices()
        self.assertEqual(indices.shape, (2,))
        self.assertTrue(np.all(indices <= 1.0))

    def test_moss_policy_uses_horizon_bonus(self) -> None:
        policy = MOSSPolicy(n_arms=2, horizon=100)
        policy.update(action=0, reward=1.0)
        policy.update(action=1, reward=0.0)
        bonuses = policy._confidence_bonus()
        self.assertEqual(bonuses.shape, (2,))
        self.assertTrue(np.all(np.isfinite(bonuses)))

    def test_cusum_ucb_resets_changed_arm(self) -> None:
        policy = CUSUMUCBPolicy(n_arms=1, threshold=0.1, drift=0.0, min_observations=2)
        policy.update(action=0, reward=0.0)
        policy.update(action=0, reward=0.0)
        policy.update(action=0, reward=5.0)
        self.assertGreaterEqual(policy.change_counts[0], 1.0)
        self.assertEqual(policy.counts[0], 1.0)

    def test_exp3_updates_action_weights(self) -> None:
        policy = EXP3Policy(n_arms=2, gamma=0.2)
        probabilities = policy.action_probabilities()
        np.testing.assert_allclose(np.sum(probabilities), 1.0)
        before = policy.weights.copy()
        action = policy.select_action(rng=np.random.default_rng(1))
        policy.update(action=action, reward=1.0)
        other = 1 - action
        self.assertEqual(policy.weights[action], before[action])
        self.assertLess(policy.weights[other], before[other])

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

    def test_sliding_window_thompson_forgets_old_binary_rewards(self) -> None:
        policy = SlidingWindowBernoulliThompsonSamplingPolicy(n_arms=1, window_size=2)
        policy.update(action=0, reward=1.0)
        policy.update(action=0, reward=0.0)
        policy.update(action=0, reward=1.0)
        self.assertEqual(policy.counts[0], 2.0)
        self.assertEqual(policy.successes[0], 1.0)
        self.assertEqual(policy.failures[0], 1.0)

    def test_discounted_thompson_discounts_posterior_counts(self) -> None:
        policy = DiscountedBernoulliThompsonSamplingPolicy(
            n_arms=1, discount_factor=0.5
        )
        policy.update(action=0, reward=1.0)
        policy.update(action=0, reward=1.0)
        self.assertLess(policy.counts[0], 2.0)
        self.assertGreater(policy.successes[0], 1.0)

    def test_successive_elimination_removes_suboptimal_arm(self) -> None:
        policy = SuccessiveEliminationPolicy(n_arms=2, delta=0.5, confidence_scale=0.01)
        for _ in range(3):
            policy.update(action=0, reward=1.0)
            policy.update(action=1, reward=0.0)
        self.assertFalse(policy.active[1])
        self.assertEqual(policy.best_arm, 0)

    def test_median_elimination_completes_phase(self) -> None:
        policy = MedianEliminationPolicy(n_arms=3, epsilon=1.0, delta=0.5)
        policy.phase_epsilon = 10.0
        policy.phase_delta = 0.5
        policy.update(action=0, reward=1.0)
        policy.update(action=1, reward=0.5)
        policy.update(action=2, reward=0.0)
        self.assertLess(np.sum(policy.active), 3)

    def test_gaussian_bayesian_ucb(self) -> None:
        policy = GaussianBayesianUCBPolicy(n_arms=2)
        action = policy.select_action(rng=np.random.default_rng(1))
        self.assertIn(action, {0, 1})

    @pytest.mark.optional
    def test_bernoulli_bayesian_ucb_uses_exact_quantile(self) -> None:
        pytest.importorskip("scipy")
        policy = BernoulliBayesianUCBPolicy(n_arms=2, quantile=0.9)
        action = policy.select_action(rng=np.random.default_rng(1))
        self.assertIn(action, {0, 1})


if __name__ == "__main__":
    unittest.main()
