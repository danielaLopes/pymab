from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from pymab import _native
from pymab._reference.registry import create_reference_policy


def _extension() -> Any:
    if not _native.native_available():
        pytest.skip("native extension has not been built in this environment")
    return importlib.import_module("pymab._pymab")


@pytest.mark.parametrize(
    ("kind", "configuration"),
    [
        ("greedy", {"n_arms": 0, "initial_value": 0.0}),
        (
            "epsilon_greedy",
            {"n_arms": 2, "initial_value": 0.0, "epsilon": 2.0},
        ),
        (
            "bernoulli_thompson_sampling",
            {"n_arms": 2, "alpha_prior": 0.0, "beta_prior": 1.0},
        ),
        (
            "lin_ucb",
            {"n_arms": 2, "n_features": 0, "alpha": 1.0, "l2": 1.0},
        ),
    ],
)
def test_invalid_policy_configuration_has_value_error_parity(
    kind: str, configuration: Mapping[str, object]
) -> None:
    with pytest.raises(ValueError):
        create_reference_policy(kind, configuration)
    with pytest.raises(ValueError):
        _extension()._NativePolicy.create(kind, json.dumps(configuration))


def test_runtime_action_and_reward_errors_have_category_parity() -> None:
    reference = create_reference_policy(
        "bernoulli_thompson_sampling",
        {"n_arms": 2, "alpha_prior": 1.0, "beta_prior": 1.0},
    )
    native = _extension()._NativePolicy.create(
        "bernoulli_thompson_sampling",
        json.dumps({"n_arms": 2, "alpha_prior": 1.0, "beta_prior": 1.0}),
    )
    with pytest.raises(ValueError, match="action"):
        cast(Any, reference).update(action=2, reward=1.0)
    with pytest.raises(ValueError, match="action"):
        native.update(2, 1.0)
    with pytest.raises(ValueError, match="binary"):
        cast(Any, reference).update(action=0, reward=0.5)
    with pytest.raises(ValueError, match="binary"):
        native.update(0, 0.5)


def test_context_shape_errors_have_value_error_parity() -> None:
    configuration = {"n_arms": 2, "n_features": 2, "alpha": 1.0, "l2": 1.0}
    reference = create_reference_policy("lin_ucb", configuration)
    native = _extension()._NativePolicy.create("lin_ucb", json.dumps(configuration))
    with pytest.raises(ValueError, match="shape"):
        cast(Any, reference).update(
            action=0,
            reward=1.0,
            context=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="expected 4 values"):
        native.update(0, 1.0, [1.0])
