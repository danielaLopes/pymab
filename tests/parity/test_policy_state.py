from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pymab.policies import (
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    MOSSPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    UCBPolicy,
)
from pymab.policies.policy import ActionValuePolicy
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
