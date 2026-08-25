from __future__ import annotations

import importlib
import json
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from pymab import _native
from pymab._reference.policies.greedy import GreedyPolicy as ReferenceGreedyPolicy
from pymab.policies import (
    BernoulliThompsonSamplingPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
    KLUCBPolicy,
    LinUCBPolicy,
    SlidingWindowUCBPolicy,
    UCBPolicy,
)
from pymab.policies._native_mixin import NativePolicyMixin, native_policy_class


def _extension() -> Any:
    if not _native.native_available():
        pytest.skip("native extension has not been built in this environment")
    return importlib.import_module("pymab._pymab")


def test_native_handle_validates_factory_and_policy_shapes() -> None:
    extension = _extension()
    native_type = extension._NativePolicy
    with pytest.raises(ValueError, match="unknown policy kind"):
        native_type.create("unknown", "{}")
    with pytest.raises(ValueError, match="fields differ"):
        native_type.create("random", '{"n_arms":2,"surprise":true}')

    classic = native_type.create("greedy", '{"n_arms":2,"initial_value":0.0}')
    assert classic.kind == "greedy"
    assert classic.is_contextual is False
    with pytest.raises(ValueError, match="outside"):
        classic.update(2, 1.0)
    with pytest.raises(TypeError, match="does not accept context"):
        classic.update(0, 1.0, [1.0, 0.0])

    contextual = native_type.create(
        "lin_ucb", '{"n_arms":2,"n_features":2,"alpha":1.0,"l2":1.0}'
    )
    assert contextual.is_contextual is True
    with pytest.raises(TypeError, match="requires context"):
        contextual.select_action(1)
    with pytest.raises(ValueError, match="expected 4 values"):
        contextual.update(0, 1.0, [1.0, 0.0, 1.0])


def test_public_native_wrapper_exposes_read_only_state_and_fresh_clone() -> None:
    _extension()
    policy = cast(
        NativePolicyMixin,
        GreedyPolicy(n_arms=2, initial_value=0.25),
    )
    assert policy.backend == "rust"
    assert isinstance(policy.configuration, MappingProxyType)
    policy.update(action=0, reward=1.0)
    assert policy.recommend_action() == 0
    assert policy.estimated_state_bytes() > 0
    estimates = cast(np.ndarray[Any, Any], policy.estimates)
    assert not estimates.flags.writeable

    clone = policy.clone()
    assert isinstance(clone, GreedyPolicy)
    assert clone.step == 0
    np.testing.assert_array_equal(clone.estimates, [0.25, 0.25])
    clone.update(action=1, reward=2.0)
    np.testing.assert_array_equal(policy.counts, [1.0, 0.0])


def test_direct_selection_consumes_numpy_rng_but_replays_equal_generators() -> None:
    _extension()
    first = GreedyPolicy(n_arms=3)
    second = GreedyPolicy(n_arms=3)
    first_rng = np.random.default_rng(44)
    second_rng = np.random.default_rng(44)
    assert first.select_action(rng=first_rng) == second.select_action(rng=second_rng)

    contextual = LinUCBPolicy(n_arms=2, n_features=2)
    context = np.eye(2)
    assert contextual.select_action(context=context, rng=np.random.default_rng(2)) in {
        0,
        1,
    }


def test_wrapper_factory_returns_reference_type_when_native_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "native_available", lambda: False)
    selected = native_policy_class(
        "greedy", ReferenceGreedyPolicy, module="pymab.policies.greedy"
    )
    assert selected is ReferenceGreedyPolicy


def test_native_configuration_round_trips_canonically() -> None:
    extension = _extension()
    config = {"n_arms": 2, "initial_value": 0.5}
    policy = extension._NativePolicy.create("greedy", json.dumps(config))
    assert json.loads(policy.configuration_json()) == config


def test_public_native_wrappers_preserve_policy_hierarchy() -> None:
    _extension()
    assert isinstance(EpsilonGreedyPolicy(n_arms=2), GreedyPolicy)
    assert isinstance(KLUCBPolicy(n_arms=2), UCBPolicy)
    assert isinstance(SlidingWindowUCBPolicy(n_arms=2), UCBPolicy)
    assert isinstance(
        DiscountedBernoulliThompsonSamplingPolicy(n_arms=2),
        BernoulliThompsonSamplingPolicy,
    )
