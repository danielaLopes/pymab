from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist
from typing import cast

import numpy as np
import pytest

from pymab.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
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
    PageHinkleyUCBPolicy,
    RandomPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
    SoftmaxPolicy,
    SuccessiveEliminationPolicy,
    UCBPolicy,
)
from pymab.policies.policy import ActionValuePolicy, ContextualPolicy, Policy
from tests.parity.conftest import load_policy_fixture, validate_policy_fixture


def _complete_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_kind": "greedy",
        "config": {"n_arms": 2},
        "updates": [{"action": 0, "reward": 1.0}],
        "checkpoints": [],
        "recommendation": 0,
        "reset_state": {},
        "expected_error": None,
    }


def test_policy_fixture_loader_accepts_the_complete_contract() -> None:
    assert validate_policy_fixture(_complete_fixture())["policy_kind"] == "greedy"


def test_python_action_value_state_uses_the_shared_fixture_shape() -> None:
    policy = GreedyPolicy(n_arms=2, initial_value=0.5)
    policy.update(action=1, reward=1.0)

    assert policy._parity_state() == {
        "step": 1,
        "total_reward": 1.0,
        "counts": [0.0, 1.0],
        "estimates": [0.5, 1.0],
    }


def _number(config: Mapping[str, object], name: str) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _integer(config: Mapping[str, object], name: str) -> int:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _numbers(value: object, *, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only numbers")
        result.append(float(item))
    return result


def _boolean(config: Mapping[str, object], name: str) -> bool:
    value = config[name]
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _string(config: Mapping[str, object], name: str) -> str:
    value = config[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _build_basic_policy(fixture: Mapping[str, object]) -> ActionValuePolicy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    if kind == "random":
        return RandomPolicy(n_arms=n_arms)
    initial_value = _number(config, "initial_value")
    if kind == "greedy":
        return GreedyPolicy(n_arms=n_arms, initial_value=initial_value)
    if kind == "epsilon_greedy":
        return EpsilonGreedyPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            epsilon=_number(config, "epsilon"),
        )
    if kind == "decaying_epsilon_greedy":
        return DecayingEpsilonGreedyPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            initial_epsilon=_number(config, "initial_epsilon"),
            min_epsilon=_number(config, "min_epsilon"),
            decay_rate=_number(config, "decay_rate"),
        )
    if kind == "softmax":
        return SoftmaxPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            temperature=_number(config, "temperature"),
        )
    raise AssertionError(f"unsupported basic fixture {kind}")


@pytest.mark.parametrize(
    "fixture_name",
    [
        "random.json",
        "greedy.json",
        "epsilon_greedy.json",
        "decaying_epsilon_greedy.json",
        "softmax.json",
    ],
    ids=lambda value: Path(value).stem,
)
def test_basic_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    policy = _build_basic_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    for update in updates:
        policy.update(
            action=_integer(update, "action"),
            reward=_number(update, "reward"),
        )

    checkpoints = cast(list[Mapping[str, object]], fixture["checkpoints"])
    final = checkpoints[-1]
    assert final["after_update"] == len(updates)
    assert policy._parity_state() == final["state"]
    assert policy.recommend_action() == fixture["recommendation"]

    scores = cast(Mapping[str, object], final["scores"])
    if isinstance(policy, DecayingEpsilonGreedyPolicy):
        assert policy.epsilon == pytest.approx(scores["epsilon"])
    if isinstance(policy, SoftmaxPolicy):
        np.testing.assert_allclose(
            policy.action_probabilities(),
            _numbers(scores["probabilities"], name="probabilities"),
            rtol=1e-15,
            atol=1e-15,
        )

    policy.reset()
    assert policy._parity_state() == fixture["reset_state"]


def _build_ucb_policy(fixture: Mapping[str, object]) -> ActionValuePolicy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    initial_value = _number(config, "initial_value")
    c = _number(config, "c")
    if kind == "ucb":
        return UCBPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            reward_scale=_number(config, "reward_scale"),
        )
    if kind == "kl_ucb":
        return KLUCBPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            c=c,
            tolerance=_number(config, "tolerance"),
            max_iterations=_integer(config, "max_iterations"),
        )
    if kind == "moss":
        return MOSSPolicy(
            n_arms=n_arms,
            initial_value=initial_value,
            horizon=_integer(config, "horizon"),
            c=c,
            reward_scale=_number(config, "reward_scale"),
        )
    raise AssertionError(f"unsupported UCB fixture {kind}")


@pytest.mark.parametrize(
    "fixture_name",
    ["ucb.json", "kl_ucb.json", "moss.json"],
    ids=lambda value: Path(value).stem,
)
def test_ucb_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    policy = _build_ucb_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    for update in updates:
        policy.update(
            action=_integer(update, "action"),
            reward=_number(update, "reward"),
        )

    checkpoints = cast(list[Mapping[str, object]], fixture["checkpoints"])
    final = checkpoints[-1]
    assert policy._parity_state() == final["state"]
    assert policy.recommend_action() == fixture["recommendation"]
    scores = cast(Mapping[str, object], final["scores"])
    if isinstance(policy, KLUCBPolicy):
        actual_scores = policy.indices()
        expected_scores = scores["indices"]
    elif isinstance(policy, (UCBPolicy, MOSSPolicy)):
        actual_scores = policy._confidence_bonus()
        expected_scores = scores["bonuses"]
    else:
        raise AssertionError("unexpected UCB policy type")
    np.testing.assert_allclose(
        actual_scores,
        _numbers(expected_scores, name="UCB scores"),
        rtol=1e-12,
        atol=1e-12,
    )

    policy.reset()
    assert policy._parity_state() == fixture["reset_state"]


def _build_posterior_policy(fixture: Mapping[str, object]) -> Policy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    if kind == "gradient_bandit":
        return GradientBanditPolicy(
            n_arms=n_arms,
            learning_rate=_number(config, "learning_rate"),
            use_baseline=_boolean(config, "use_baseline"),
        )
    if kind == "bernoulli_thompson_sampling":
        return BernoulliThompsonSamplingPolicy(
            n_arms=n_arms,
            alpha_prior=_number(config, "alpha_prior"),
            beta_prior=_number(config, "beta_prior"),
        )
    if kind == "gaussian_thompson_sampling":
        return GaussianThompsonSamplingPolicy(
            n_arms=n_arms,
            prior_mean=_number(config, "prior_mean"),
            prior_precision=_number(config, "prior_precision"),
            reward_precision=_number(config, "reward_precision"),
        )
    if kind == "bernoulli_bayesian_ucb":
        return BernoulliBayesianUCBPolicy(
            n_arms=n_arms,
            alpha_prior=_number(config, "alpha_prior"),
            beta_prior=_number(config, "beta_prior"),
            quantile=_number(config, "quantile"),
        )
    if kind == "gaussian_bayesian_ucb":
        return GaussianBayesianUCBPolicy(
            n_arms=n_arms,
            prior_mean=_number(config, "prior_mean"),
            prior_precision=_number(config, "prior_precision"),
            reward_precision=_number(config, "reward_precision"),
            quantile=_number(config, "quantile"),
        )
    raise AssertionError(f"unsupported posterior fixture {kind}")


def _posterior_state(policy: Policy) -> dict[str, object]:
    if isinstance(policy, GradientBanditPolicy):
        return {
            "step": policy.step,
            "average_reward": policy.average_reward,
            "preferences": policy.preferences.tolist(),
            "probabilities": policy.probabilities.tolist(),
        }
    if not isinstance(policy, ActionValuePolicy):
        raise AssertionError("posterior policy must expose action-value state")
    state = policy._parity_state()
    if isinstance(
        policy, (BernoulliThompsonSamplingPolicy, BernoulliBayesianUCBPolicy)
    ):
        state.update(
            successes=policy.successes.tolist(),
            failures=policy.failures.tolist(),
        )
    if isinstance(policy, (GaussianThompsonSamplingPolicy, GaussianBayesianUCBPolicy)):
        state.update(
            means=policy.means.tolist(),
            precisions=policy.precisions.tolist(),
        )
    return state


@pytest.mark.parametrize(
    "fixture_name",
    [
        "gradient.json",
        "bernoulli_thompson.json",
        "gaussian_thompson.json",
        "bernoulli_bayesian_ucb.json",
        "gaussian_bayesian_ucb.json",
    ],
    ids=lambda value: Path(value).stem,
)
def test_posterior_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    policy = _build_posterior_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    for update in updates:
        policy.update(
            action=_integer(update, "action"),
            reward=_number(update, "reward"),
        )

    checkpoints = cast(list[Mapping[str, object]], fixture["checkpoints"])
    final = checkpoints[-1]
    assert _posterior_state(policy) == final["state"]
    assert policy.recommend_action() == fixture["recommendation"]
    scores = cast(Mapping[str, object], final["scores"])
    if isinstance(policy, BernoulliBayesianUCBPolicy):
        beta_distribution = pytest.importorskip("scipy.stats").beta
        actual_scores = beta_distribution.ppf(
            policy.quantile,
            policy.alpha_prior + policy.successes,
            policy.beta_prior + policy.failures,
        )
        expected_scores = scores["upper_bounds"]
        np.testing.assert_allclose(
            actual_scores,
            _numbers(expected_scores, name="upper bounds"),
            rtol=1e-12,
            atol=1e-12,
        )
    if isinstance(policy, GaussianBayesianUCBPolicy):
        actual_scores = policy.means + NormalDist().inv_cdf(policy.quantile) / np.sqrt(
            policy.precisions
        )
        np.testing.assert_allclose(
            actual_scores,
            _numbers(scores["upper_bounds"], name="upper bounds"),
            rtol=1e-12,
            atol=1e-12,
        )

    policy.reset()
    assert _posterior_state(policy) == fixture["reset_state"]


def _build_exploration_policy(fixture: Mapping[str, object]) -> ActionValuePolicy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    if kind == "exp3":
        return EXP3Policy(
            n_arms=n_arms,
            gamma=_number(config, "gamma"),
            learning_rate=_number(config, "learning_rate"),
        )
    if kind == "successive_elimination":
        return SuccessiveEliminationPolicy(
            n_arms=n_arms,
            delta=_number(config, "delta"),
            confidence_scale=_number(config, "confidence_scale"),
        )
    if kind == "median_elimination":
        return MedianEliminationPolicy(
            n_arms=n_arms,
            epsilon=_number(config, "epsilon"),
            delta=_number(config, "delta"),
        )
    raise AssertionError(f"unsupported exploration fixture {kind}")


def _exploration_state(policy: ActionValuePolicy) -> dict[str, object]:
    state = policy._parity_state()
    if isinstance(policy, EXP3Policy):
        state.update(
            log_weights=policy.log_weights.tolist(),
            last_probabilities=policy.last_probabilities.tolist(),
        )
    if isinstance(policy, SuccessiveEliminationPolicy):
        state.update(active=policy.active.tolist())
    if isinstance(policy, MedianEliminationPolicy):
        state.update(
            active=policy.active.tolist(),
            phase_counts=policy.phase_counts.tolist(),
            phase_sums=policy.phase_sums.tolist(),
            phase_epsilon=policy.phase_epsilon,
            phase_delta=policy.phase_delta,
        )
    return state


@pytest.mark.parametrize(
    "fixture_name",
    ["exp3.json", "successive_elimination.json", "median_elimination.json"],
    ids=lambda value: Path(value).stem,
)
def test_exploration_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    policy = _build_exploration_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    completed_updates = 0
    for update in updates:
        repeat_value = update.get("repeat", 1)
        if isinstance(repeat_value, bool) or not isinstance(repeat_value, int):
            raise TypeError("repeat must be an integer")
        for _ in range(repeat_value):
            policy.update(
                action=_integer(update, "action"),
                reward=_number(update, "reward"),
            )
            completed_updates += 1

    checkpoints = cast(list[Mapping[str, object]], fixture["checkpoints"])
    final = checkpoints[-1]
    assert final["after_update"] == completed_updates
    assert _exploration_state(policy) == final["state"]
    assert policy.recommend_action() == fixture["recommendation"]
    scores = cast(Mapping[str, object], final["scores"])
    if isinstance(policy, EXP3Policy):
        np.testing.assert_allclose(
            policy.action_probabilities(),
            _numbers(scores["probabilities"], name="probabilities"),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            policy.weights,
            _numbers(scores["weights"], name="weights"),
            rtol=1e-12,
            atol=1e-12,
        )
    if isinstance(policy, SuccessiveEliminationPolicy):
        np.testing.assert_allclose(
            policy._confidence_radii(),
            _numbers(scores["confidence_radii"], name="confidence radii"),
            rtol=1e-12,
            atol=1e-12,
        )
    if isinstance(policy, MedianEliminationPolicy):
        assert policy._phase_quota() == scores["phase_quota"]

    policy.reset()
    assert _exploration_state(policy) == fixture["reset_state"]


def _build_adaptive_policy(fixture: Mapping[str, object]) -> ActionValuePolicy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    common = {
        "n_arms": _integer(config, "n_arms"),
    }
    if kind == "sliding_window_ucb":
        return SlidingWindowUCBPolicy(
            **common,
            initial_value=_number(config, "initial_value"),
            c=_number(config, "c"),
            reward_scale=_number(config, "reward_scale"),
            window_size=_integer(config, "window_size"),
        )
    if kind == "discounted_ucb":
        return DiscountedUCBPolicy(
            **common,
            initial_value=_number(config, "initial_value"),
            c=_number(config, "c"),
            reward_scale=_number(config, "reward_scale"),
            discount_factor=_number(config, "discount_factor"),
        )
    if kind == "sliding_window_bernoulli_thompson_sampling":
        return SlidingWindowBernoulliThompsonSamplingPolicy(
            **common,
            alpha_prior=_number(config, "alpha_prior"),
            beta_prior=_number(config, "beta_prior"),
            window_size=_integer(config, "window_size"),
        )
    if kind == "discounted_bernoulli_thompson_sampling":
        return DiscountedBernoulliThompsonSamplingPolicy(
            **common,
            alpha_prior=_number(config, "alpha_prior"),
            beta_prior=_number(config, "beta_prior"),
            discount_factor=_number(config, "discount_factor"),
        )
    initial_value = _number(config, "initial_value")
    c = _number(config, "c")
    reward_scale = _number(config, "reward_scale")
    threshold = _number(config, "threshold")
    drift = _number(config, "drift")
    min_observations = _integer(config, "min_observations")
    if kind == "change_point_ucb":
        return ChangePointUCBPolicy(
            n_arms=common["n_arms"],
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
            detector=_string(config, "detector"),
            threshold=threshold,
            drift=drift,
            min_observations=min_observations,
        )
    if kind == "cusum_ucb":
        return CUSUMUCBPolicy(
            n_arms=common["n_arms"],
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
            threshold=threshold,
            drift=drift,
            min_observations=min_observations,
        )
    if kind == "page_hinkley_ucb":
        return PageHinkleyUCBPolicy(
            n_arms=common["n_arms"],
            initial_value=initial_value,
            c=c,
            reward_scale=reward_scale,
            threshold=threshold,
            drift=drift,
            min_observations=min_observations,
        )
    raise AssertionError(f"unsupported adaptive fixture {kind}")


def _adaptive_state(policy: ActionValuePolicy) -> dict[str, object]:
    state = policy._parity_state()
    if isinstance(policy, SlidingWindowUCBPolicy):
        state["history_len"] = len(policy._history)
    if isinstance(policy, DiscountedUCBPolicy):
        state.update(
            discounted_counts=policy.discounted_counts.tolist(),
            discounted_sums=policy.discounted_sums.tolist(),
        )
    if isinstance(policy, SlidingWindowBernoulliThompsonSamplingPolicy):
        state.update(
            successes=policy.successes.tolist(),
            failures=policy.failures.tolist(),
            history_len=len(policy._history),
        )
    if isinstance(policy, DiscountedBernoulliThompsonSamplingPolicy):
        state.update(
            successes=policy.successes.tolist(),
            failures=policy.failures.tolist(),
        )
    if isinstance(policy, ChangePointUCBPolicy):
        state.update(
            detector_counts=policy.detector_counts.tolist(),
            detector_means=policy.detector_means.tolist(),
            positive_cusum=policy.positive_cusum.tolist(),
            negative_cusum=policy.negative_cusum.tolist(),
            ph_cumulative=policy.ph_cumulative.tolist(),
            ph_minimum=policy.ph_minimum.tolist(),
            change_counts=policy.change_counts.tolist(),
        )
    return state


@pytest.mark.parametrize(
    "fixture_name",
    [
        "sliding_window_ucb.json",
        "discounted_ucb.json",
        "sliding_window_bernoulli_thompson.json",
        "discounted_bernoulli_thompson.json",
        "change_point_ucb.json",
        "cusum_ucb.json",
        "page_hinkley_ucb.json",
    ],
    ids=lambda value: Path(value).stem,
)
def test_adaptive_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    policy = _build_adaptive_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    for update in updates:
        policy.update(
            action=_integer(update, "action"),
            reward=_number(update, "reward"),
        )

    final = cast(list[Mapping[str, object]], fixture["checkpoints"])[-1]
    assert final["after_update"] == len(updates)
    assert _adaptive_state(policy) == final["state"]
    assert policy.recommend_action() == fixture["recommendation"]
    if isinstance(policy, (SlidingWindowUCBPolicy, DiscountedUCBPolicy)):
        scores = cast(Mapping[str, object], final["scores"])
        np.testing.assert_allclose(
            policy._confidence_bonus(),
            _numbers(scores["bonuses"], name="bonuses"),
            rtol=1e-12,
            atol=1e-12,
        )

    policy.reset()
    assert _adaptive_state(policy) == fixture["reset_state"]


def _build_contextual_policy(fixture: Mapping[str, object]) -> ContextualPolicy:
    kind = cast(str, fixture["policy_kind"])
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    n_features = _integer(config, "n_features")
    if kind == "linear_epsilon_greedy":
        return LinearEpsilonGreedyPolicy(
            n_arms=n_arms,
            n_features=n_features,
            epsilon=_number(config, "epsilon"),
            learning_rate=_number(config, "learning_rate"),
        )
    if kind == "lin_ucb":
        return LinUCBPolicy(
            n_arms=n_arms,
            n_features=n_features,
            alpha=_number(config, "alpha"),
            l2=_number(config, "l2"),
        )
    if kind == "linear_thompson_sampling":
        return LinearThompsonSamplingPolicy(
            n_arms=n_arms,
            n_features=n_features,
            exploration_scale=_number(config, "exploration_scale"),
            l2=_number(config, "l2"),
        )
    if kind == "logistic_contextual_bandit":
        return LogisticContextualBanditPolicy(
            n_arms=n_arms,
            n_features=n_features,
            epsilon=_number(config, "epsilon"),
            learning_rate=_number(config, "learning_rate"),
            l2=_number(config, "l2"),
        )
    raise AssertionError(f"unsupported contextual fixture {kind}")


def _context(value: object, *, n_arms: int, n_features: int) -> np.ndarray:
    return np.asarray(_numbers(value, name="context"), dtype=float).reshape(
        n_arms, n_features
    )


def _contextual_state(policy: ContextualPolicy) -> dict[str, object]:
    if isinstance(policy, (LinearEpsilonGreedyPolicy, LogisticContextualBanditPolicy)):
        return {"theta": policy.theta.reshape(-1).tolist()}
    if isinstance(policy, (LinUCBPolicy, LinearThompsonSamplingPolicy)):
        return {
            "a": policy.a.reshape(-1).tolist(),
            "b": policy.b.reshape(-1).tolist(),
        }
    raise AssertionError("unexpected contextual policy")


@pytest.mark.parametrize(
    "fixture_name",
    [
        "linear_epsilon_greedy.json",
        "lin_ucb.json",
        "linear_thompson_sampling.json",
        "logistic_contextual_bandit.json",
    ],
    ids=lambda value: Path(value).stem,
)
def test_contextual_policy_matches_shared_state_fixture(fixture_name: str) -> None:
    fixture = load_policy_fixture(
        Path(__file__).parents[1] / "fixtures" / "policies" / fixture_name
    )
    config = cast(Mapping[str, object], fixture["config"])
    n_arms = _integer(config, "n_arms")
    n_features = _integer(config, "n_features")
    policy = _build_contextual_policy(fixture)
    updates = cast(list[Mapping[str, object]], fixture["updates"])
    for update in updates:
        policy.update(
            action=_integer(update, "action"),
            reward=_number(update, "reward"),
            context=_context(update["context"], n_arms=n_arms, n_features=n_features),
        )

    final = cast(list[Mapping[str, object]], fixture["checkpoints"])[-1]
    assert final["after_update"] == len(updates)
    assert _contextual_state(policy) == final["state"]
    scores = cast(Mapping[str, object], final["scores"])
    score_context = _context(scores["context"], n_arms=n_arms, n_features=n_features)
    if isinstance(policy, LinearEpsilonGreedyPolicy):
        actual_scores = policy.scores(score_context)
    elif isinstance(policy, LinUCBPolicy):
        actual_scores = policy.upper_confidence_bounds(score_context)
    elif isinstance(policy, LinearThompsonSamplingPolicy):
        actual_scores = np.asarray(
            [
                np.linalg.solve(policy.a[arm], policy.b[arm]) @ score_context[arm]
                for arm in range(n_arms)
            ]
        )
    elif isinstance(policy, LogisticContextualBanditPolicy):
        actual_scores = policy.predicted_probabilities(score_context)
    else:
        raise AssertionError("unexpected contextual policy")
    np.testing.assert_allclose(
        actual_scores,
        _numbers(scores["values"], name="contextual scores"),
        rtol=1e-12,
        atol=1e-12,
    )
    recommendation = cast(Mapping[str, object], fixture["recommendation"])
    recommendation_context = _context(
        recommendation["context"], n_arms=n_arms, n_features=n_features
    )
    assert policy.recommend_action(context=recommendation_context) == _integer(
        recommendation, "action"
    )

    policy.reset()
    assert _contextual_state(policy) == fixture["reset_state"]


@pytest.mark.parametrize("field", sorted(_complete_fixture()))
def test_policy_fixture_loader_rejects_missing_fields(field: str) -> None:
    fixture = _complete_fixture()
    fixture.pop(field)
    with pytest.raises(ValueError, match="missing"):
        validate_policy_fixture(fixture)


def test_policy_fixture_loader_rejects_unknown_fields() -> None:
    fixture = _complete_fixture()
    fixture["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_policy_fixture(fixture)
