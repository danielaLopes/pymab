from __future__ import annotations

import numpy as np
import pytest

from pymab.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    ChangePointUCBPolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    EpsilonGreedyPolicy,
    EXP3Policy,
    GaussianBayesianUCBPolicy,
    GaussianThompsonSamplingPolicy,
    GradientBanditPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
    MedianEliminationPolicy,
    MOSSPolicy,
    RandomPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    SuccessiveEliminationPolicy,
    UCBPolicy,
)
from pymab.policies.policy import choose_argmax, softmax, validate_probability


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GreedyPolicy(n_arms=0),
        lambda: GreedyPolicy(n_arms=2.5),
        lambda: EpsilonGreedyPolicy(n_arms=2, epsilon=-1),
        lambda: EpsilonGreedyPolicy(n_arms=2, epsilon=np.nan),
        lambda: DecayingEpsilonGreedyPolicy(
            n_arms=2, min_epsilon=0.8, initial_epsilon=0.2
        ),
        lambda: DecayingEpsilonGreedyPolicy(n_arms=2, decay_rate=-1),
        lambda: SoftmaxPolicy(n_arms=2, temperature=0),
        lambda: GradientBanditPolicy(n_arms=2, learning_rate=0),
        lambda: GradientBanditPolicy(n_arms=2, learning_rate=np.nan),
        lambda: UCBPolicy(n_arms=2, c=0),
        lambda: KLUCBPolicy(n_arms=2, tolerance=0),
        lambda: KLUCBPolicy(n_arms=2, max_iterations=0),
        lambda: MOSSPolicy(n_arms=2, horizon=0),
        lambda: MOSSPolicy(n_arms=3, horizon=2),
        lambda: SlidingWindowUCBPolicy(n_arms=2, window_size=0),
        lambda: DiscountedUCBPolicy(n_arms=2, discount_factor=1),
        lambda: ChangePointUCBPolicy(n_arms=2, detector="unknown"),
        lambda: ChangePointUCBPolicy(n_arms=2, threshold=0),
        lambda: ChangePointUCBPolicy(n_arms=2, drift=-1),
        lambda: ChangePointUCBPolicy(n_arms=2, min_observations=0),
        lambda: EXP3Policy(n_arms=2, learning_rate=0),
        lambda: EXP3Policy(n_arms=2, gamma=0),
        lambda: EXP3Policy(n_arms=2, learning_rate=2),
        lambda: BernoulliThompsonSamplingPolicy(n_arms=2, alpha_prior=0),
        lambda: GaussianThompsonSamplingPolicy(n_arms=2, prior_precision=0),
        lambda: SlidingWindowBernoulliThompsonSamplingPolicy(n_arms=2, window_size=0),
        lambda: DiscountedBernoulliThompsonSamplingPolicy(n_arms=2, discount_factor=0),
        lambda: BernoulliBayesianUCBPolicy(n_arms=2, quantile=1),
        lambda: GaussianBayesianUCBPolicy(n_arms=2, reward_precision=0),
        lambda: SuccessiveEliminationPolicy(n_arms=2, delta=0),
        lambda: SuccessiveEliminationPolicy(n_arms=2, confidence_scale=0),
        lambda: MedianEliminationPolicy(n_arms=2, epsilon=0),
        lambda: MedianEliminationPolicy(n_arms=2, delta=0),
        lambda: LinearEpsilonGreedyPolicy(n_arms=2, n_features=2, learning_rate=0),
        lambda: LinUCBPolicy(n_arms=2, n_features=2, alpha=0),
        lambda: LinUCBPolicy(n_arms=2, n_features=2, l2=0),
        lambda: LinearThompsonSamplingPolicy(
            n_arms=2, n_features=2, exploration_scale=0
        ),
        lambda: LogisticContextualBanditPolicy(n_arms=2, n_features=2, l2=-1),
    ],
)
def test_constructor_validation(constructor) -> None:
    with pytest.raises((ValueError, TypeError)):
        constructor()


def test_policy_helpers_and_action_validation() -> None:
    with pytest.raises(ValueError):
        validate_probability(2, name="value")
    with pytest.raises(ValueError):
        softmax(np.array([1.0]), temperature=0)
    with pytest.raises(ValueError):
        softmax(np.array([np.nan]))
    probabilities = softmax(np.array([1000.0, 1000.0]))
    np.testing.assert_allclose(probabilities, [0.5, 0.5])
    choices = {
        choose_argmax(np.array([1.0, 1.0]), np.random.default_rng(seed))
        for seed in range(10)
    }
    assert choices == {0, 1}
    with pytest.raises(ValueError):
        GreedyPolicy(n_arms=2).update(action=2, reward=1)
    with pytest.raises(ValueError):
        GreedyPolicy(n_arms=2).update(action=0, reward=np.nan)


def test_clone_resets_state_without_sharing_arrays() -> None:
    policy = EpsilonGreedyPolicy(n_arms=2)
    policy.update(action=0, reward=1)
    clone = policy.clone()
    assert clone.step == 0
    clone.estimates[0] = 99
    assert policy.estimates[0] == 1


def test_windowed_policies_expire_observations_by_global_time() -> None:
    ucb = SlidingWindowUCBPolicy(n_arms=2, window_size=2)
    ucb.update(action=0, reward=1)
    ucb.update(action=1, reward=0)
    ucb.update(action=1, reward=0)
    np.testing.assert_array_equal(ucb.counts, [0, 2])
    np.testing.assert_array_equal(ucb.estimates, [0, 0])

    thompson = SlidingWindowBernoulliThompsonSamplingPolicy(n_arms=2, window_size=2)
    thompson.update(action=0, reward=1)
    thompson.update(action=1, reward=0)
    thompson.update(action=1, reward=0)
    np.testing.assert_array_equal(thompson.counts, [0, 2])
    np.testing.assert_array_equal(thompson.successes, [0, 0])
    np.testing.assert_array_equal(thompson.failures, [0, 2])


def test_exp3_remains_finite_under_long_concentrated_updates() -> None:
    policy = EXP3Policy(n_arms=3, gamma=1e-6, learning_rate=1.0)
    rng = np.random.default_rng(8)
    for _ in range(20_000):
        action = policy.select_action(rng=rng)
        policy.update(action=action, reward=1.0)
    probabilities = policy.action_probabilities()
    assert np.all(np.isfinite(policy.log_weights))
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0)
    assert np.sum(probabilities) == pytest.approx(1.0)


def test_ucb_reward_scale_controls_confidence_width() -> None:
    narrow = UCBPolicy(n_arms=1, reward_scale=1.0)
    wide = UCBPolicy(n_arms=1, reward_scale=2.0)
    narrow.update(action=0, reward=0)
    wide.update(action=0, reward=0)
    assert wide._confidence_bonus()[0] == pytest.approx(
        2 * narrow._confidence_bonus()[0]
    )


def test_all_classic_policies_select_valid_actions_and_repr() -> None:
    policies = [
        GreedyPolicy(n_arms=2),
        RandomPolicy(n_arms=2),
        EpsilonGreedyPolicy(n_arms=2, epsilon=1),
        DecayingEpsilonGreedyPolicy(n_arms=2),
        SoftmaxPolicy(n_arms=2),
        GradientBanditPolicy(n_arms=2),
        UCBPolicy(n_arms=2),
        MOSSPolicy(n_arms=2, horizon=10),
        SlidingWindowUCBPolicy(n_arms=2),
        DiscountedUCBPolicy(n_arms=2),
        EXP3Policy(n_arms=2),
        GaussianThompsonSamplingPolicy(n_arms=2),
        GaussianBayesianUCBPolicy(n_arms=2),
        SuccessiveEliminationPolicy(n_arms=2),
        MedianEliminationPolicy(n_arms=2, epsilon=1, delta=0.5),
    ]
    for index, policy in enumerate(policies):
        action = policy.select_action(rng=np.random.default_rng(index))
        assert action in {0, 1}
        assert type(policy).__name__.split("Policy")[0] in repr(policy)


def test_gaussian_posterior_updates_are_exact() -> None:
    thompson = GaussianThompsonSamplingPolicy(
        n_arms=1, prior_mean=0, prior_precision=1, reward_precision=1
    )
    thompson.update(action=0, reward=2)
    assert thompson.means[0] == 1
    assert thompson.precisions[0] == 2
    bayesian = GaussianBayesianUCBPolicy(
        n_arms=1, prior_mean=0, prior_precision=1, reward_precision=1
    )
    bayesian.update(action=0, reward=2)
    assert bayesian.means[0] == 1
    assert bayesian.precisions[0] == 2


@pytest.mark.optional
def test_bernoulli_bayesian_updates_and_selects() -> None:
    pytest.importorskip("scipy")
    policy = BernoulliBayesianUCBPolicy(n_arms=2)
    policy.update(action=0, reward=1)
    policy.update(action=1, reward=0)
    assert policy.successes[0] == 1
    assert policy.failures[1] == 1
    assert policy.select_action(rng=np.random.default_rng(1)) == 0
    with pytest.raises(ValueError, match="binary"):
        policy.update(action=0, reward=0.5)


def test_thompson_variants_update_and_select() -> None:
    bernoulli = BernoulliThompsonSamplingPolicy(n_arms=2)
    bernoulli.update(action=0, reward=1)
    bernoulli.update(action=1, reward=0)
    assert bernoulli.select_action(rng=np.random.default_rng(2)) in {0, 1}
    sliding = SlidingWindowBernoulliThompsonSamplingPolicy(n_arms=2, window_size=2)
    sliding.update(action=0, reward=1)
    sliding.update(action=0, reward=0)
    assert "window_size=2" in repr(sliding)
    discounted = DiscountedBernoulliThompsonSamplingPolicy(
        n_arms=2, discount_factor=0.5
    )
    discounted.update(action=0, reward=1)
    discounted.update(action=1, reward=0)
    assert "discount_factor=0.5" in repr(discounted)


def test_pure_exploration_single_arm_and_phase_paths() -> None:
    successive = SuccessiveEliminationPolicy(n_arms=1)
    assert successive.select_action(rng=np.random.default_rng(1)) == 0
    assert successive.best_arm == 0
    assert successive.recommend_action() == 0
    median = MedianEliminationPolicy(n_arms=1)
    assert median.select_action(rng=np.random.default_rng(1)) == 0
    assert median.best_arm == 0
    assert median.recommend_action() == 0
    multi = MedianEliminationPolicy(n_arms=2, epsilon=1, delta=0.5)
    multi.phase_epsilon = 10
    multi.update(action=0, reward=1)
    multi.update(action=1, reward=0)
    assert np.sum(multi.active) == 1


def test_contextual_validation_and_exploration_paths() -> None:
    context = np.eye(2)
    epsilon = LinearEpsilonGreedyPolicy(n_arms=2, n_features=2, epsilon=1)
    assert epsilon.select_action(context=context, rng=np.random.default_rng(1)) in {
        0,
        1,
    }
    logistic = LogisticContextualBanditPolicy(n_arms=2, n_features=2, epsilon=1)
    assert logistic.select_action(context=context, rng=np.random.default_rng(1)) in {
        0,
        1,
    }
    with pytest.raises(ValueError, match="shape"):
        epsilon.select_action(context=np.ones((1, 2)), rng=np.random.default_rng(1))
