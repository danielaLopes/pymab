from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import pytest

from pymab import _native
from pymab.distributions import BernoulliReward, UniformReward
from pymab.environments import (
    BanditEnvironment,
    FixedContextProvider,
    LinearContextualEnvironment,
    LogisticContextualEnvironment,
    ProbabilityDrift,
)


def _extension() -> Any:
    if not _native.native_available():
        pytest.skip("native extension has not been built in this environment")
    return importlib.import_module("pymab._pymab")


def _native_environment(configuration: dict[str, object]) -> Any:
    return _extension()._NativeEnvironment.create(json.dumps(configuration))


def test_native_classic_environment_matches_deterministic_reference_trace() -> None:
    reference = BanditEnvironment(
        means=np.array([0.0, 0.5, 1.0]),
        reward_model=BernoulliReward(),
        dynamics=ProbabilityDrift(logit_std=0.0, epsilon=1e-9),
    )
    native = _native_environment(
        {
            "kind": "classic",
            "means": [0.0, 0.5, 1.0],
            "reward": {"kind": "bernoulli"},
            "dynamics": {
                "kind": "probability",
                "logit_std": 0.0,
                "epsilon": 1e-9,
            },
        }
    )
    assert native.contextual is False
    assert native.n_arms == 3
    assert native.n_features is None

    for step in range(3):
        reference.advance(step=step, rng=np.random.default_rng(step))
        native.advance(step, step)
        native_means = json.loads(native.state_json())["means"]
        np.testing.assert_allclose(native_means, reference.means, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            native.expected_rewards(), reference.expected_rewards()
        )


def test_native_uniform_reward_matches_zero_width_reference_exactly() -> None:
    reference = BanditEnvironment(
        means=np.array([-1.0, 2.0]),
        reward_model=UniformReward(half_width=0.0),
    )
    native = _native_environment(
        {
            "kind": "classic",
            "means": [-1.0, 2.0],
            "reward": {"kind": "uniform", "half_width": 0.0},
            "dynamics": {"kind": "stationary"},
        }
    )
    np.testing.assert_array_equal(
        native.sample_rewards(5),
        reference.sample_rewards(rng=np.random.default_rng(5)),
    )


@pytest.mark.parametrize("kind", ["linear", "logistic"])
def test_native_contextual_expected_rewards_match_reference(kind: str) -> None:
    theta = np.array([[2.0, -1.0], [0.5, 3.0]])
    context = np.array([[1.0, 2.0], [-1.0, 0.25]])
    provider = FixedContextProvider(context)
    if kind == "linear":
        reference = LinearContextualEnvironment(
            theta=theta,
            context_provider=provider,
            reward_model=UniformReward(half_width=0.0),
        )
        reward: dict[str, object] = {"kind": "uniform", "half_width": 0.0}
    else:
        reference = LogisticContextualEnvironment(
            theta=theta,
            context_provider=provider,
            reward_model=BernoulliReward(),
        )
        reward = {"kind": "bernoulli"}
    native = _native_environment(
        {
            "kind": kind,
            "n_arms": 2,
            "n_features": 2,
            "theta": theta.reshape(-1).tolist(),
            "context_provider": {
                "kind": "fixed",
                "values": context.reshape(-1).tolist(),
            },
            "reward": reward,
        }
    )

    native_context = np.asarray(native.context(9)).reshape(2, 2)
    np.testing.assert_array_equal(native_context, context)
    np.testing.assert_allclose(
        native.expected_rewards(context.reshape(-1).tolist()),
        reference.expected_rewards(context),
        rtol=1e-12,
        atol=1e-12,
    )
    if kind == "linear":
        np.testing.assert_allclose(
            native.sample_rewards(4, context.reshape(-1).tolist()),
            reference.sample_rewards(context=context, rng=np.random.default_rng(4)),
        )


def test_native_environment_factory_is_strict_and_context_methods_are_typed() -> None:
    extension = _extension()
    with pytest.raises(ValueError, match="unknown field"):
        extension._NativeEnvironment.create(
            json.dumps(
                {
                    "kind": "classic",
                    "means": [0.0],
                    "reward": {"kind": "gaussian", "std": 1.0},
                    "dynamics": {"kind": "stationary"},
                    "surprise": True,
                }
            )
        )
    classic = _native_environment(
        {
            "kind": "classic",
            "means": [0.0],
            "reward": {"kind": "gaussian", "std": 1.0},
            "dynamics": {"kind": "stationary"},
        }
    )
    with pytest.raises(TypeError, match="does not produce context"):
        classic.context(1)
    with pytest.raises(TypeError, match="does not accept context"):
        classic.expected_rewards([1.0])
