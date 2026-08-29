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
    validate_environment,
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
        "epsilon": epsilon,
        "initial_value": 0.0,
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
        ("linucb", {"alpha": 1.0, "l2": 0.0}, "l2"),
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


@pytest.mark.parametrize(
    ("lesson", "parameters", "message"),
    [
        ("epsilon-greedy", {"epsilon": 0.2, "unknown": 1}, "unknown parameters"),
        ("epsilon-greedy", {"epsilon": 0.2, "initial_value": np.inf}, "finite"),
        ("ucb", {"c": 2.0, "reward_scale": 1.0, "unknown": 1}, "unknown parameters"),
        ("gradient-bandit", {"learning_rate": 0.1, "use_baseline": 1}, "boolean"),
        ("change-point-ucb", {"detector": 1}, "text"),
        (
            "sliding-window-ucb",
            {"window_size": 2.5, "c": 2.0, "reward_scale": 1.0},
            "integer",
        ),
    ],
)
def test_catalog_parameter_types_are_rejected(
    lesson: str, parameters: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_parameters(lesson, parameters)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lesson", "environment", "message"),
    [
        ("epsilon-greedy", {"means": [0.1, 0.2, 0.3]}, "only probabilities"),
        ("epsilon-greedy", {"probabilities": [0.1, np.inf, 0.3]}, "finite"),
        ("gaussian-thompson-sampling", {"means": [0, 1, 2]}, "requires means"),
        (
            "gaussian-thompson-sampling",
            {"means": [0, 1, 2], "standardDeviation": 0},
            "positive",
        ),
        ("sliding-window-ucb", {"probabilities": [0.1, 0.2, 0.3]}, "requires phases"),
        ("sliding-window-ucb", {"phases": []}, "two to four"),
        (
            "sliding-window-ucb",
            {"phases": [{"start": 0}] * 2},
            "start and probabilities",
        ),
        (
            "sliding-window-ucb",
            {
                "phases": [
                    {"start": -1, "probabilities": [0.1, 0.2, 0.3]},
                    {"start": 2, "probabilities": [0.2, 0.3, 0.4]},
                ]
            },
            "non-negative",
        ),
        (
            "sliding-window-ucb",
            {
                "phases": [
                    {"start": 1, "probabilities": [0.1, 0.2, 0.3]},
                    {"start": 2, "probabilities": [0.2, 0.3, 0.4]},
                ]
            },
            "begin at zero",
        ),
        ("exp3", {"probabilities": [0.1, 0.2, 0.3]}, "requires rewards"),
        ("exp3", {"rewards": []}, "at least one"),
        ("linucb", {"probabilities": [0.1, 0.2, 0.3]}, "only available"),
        ("linucb", {"unexpected": 1}, "unexpected fields"),
        ("linucb", {"theta": [[0, 0, 0, 0]]}, "three portal rows"),
        ("linucb", {"theta": [[0, 0, 0]] * 3}, "four coefficients"),
        ("linucb", {"theta": [[0, 0, 0, np.inf]] * 3}, "finite"),
        (
            "linear-thompson-sampling",
            {"theta": [[0, 0, 0, 0]] * 3, "standardDeviation": 0},
            "positive",
        ),
    ],
)
def test_invalid_environment_shapes_are_rejected(
    lesson: str, environment: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_environment(lesson, "freePlay", environment)  # type: ignore[arg-type]
