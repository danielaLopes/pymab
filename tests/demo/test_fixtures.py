from __future__ import annotations

import numpy as np
import pytest
from pymab_demo.fixtures import (
    ALPHA_RANGE,
    EPSILON_MEANS,
    EPSILON_RANGE,
    FIXTURES,
    LINUCB_THETA,
    horizon_for,
    validate_parameters,
)


def test_canonical_fixture_domains() -> None:
    assert EPSILON_MEANS == (0.25, 0.5, 0.75)
    assert EPSILON_RANGE == (0.0, 1.0, 0.01)
    assert ALPHA_RANGE == (0.05, 4.0, 0.05)
    assert LINUCB_THETA.shape == (3, 4)
    assert np.isfinite(LINUCB_THETA).all()
    assert FIXTURES["epsilon-greedy"].guided_seed == 42
    assert FIXTURES["linucb"].challenge_seed == 20260824
    assert horizon_for("linucb", "guided") == 12
    assert horizon_for("linucb", "freePlay") == 20


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.35, 1.0])
def test_epsilon_range_values_are_accepted(epsilon: float) -> None:
    assert validate_parameters("epsilon-greedy", {"epsilon": epsilon}) == {
        "epsilon": epsilon
    }


@pytest.mark.parametrize("alpha", [0.05, 0.1, 1.0, 3.0, 4.0])
def test_alpha_range_values_are_accepted(alpha: float) -> None:
    assert validate_parameters("linucb", {"alpha": alpha, "l2": 1}) == {
        "alpha": alpha,
        "l2": 1.0,
    }


@pytest.mark.parametrize(
    ("lesson", "parameters", "message"),
    [
        ("epsilon-greedy", {"epsilon": -0.01}, "epsilon"),
        ("epsilon-greedy", {"epsilon": 0.305}, "epsilon"),
        ("linucb", {"alpha": 0.0}, "alpha"),
        ("linucb", {"alpha": 1.03}, "alpha"),
        ("linucb", {"alpha": 1.0, "l2": 2.0}, "l2"),
    ],
)
def test_invalid_parameters_are_rejected(
    lesson: str, parameters: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_parameters(lesson, parameters)  # type: ignore[arg-type]


def test_non_numeric_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="numeric"):
        validate_parameters("epsilon-greedy", {"epsilon": True})
