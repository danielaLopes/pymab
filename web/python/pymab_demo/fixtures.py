"""Canonical lesson fixtures shared by the browser and CPython tests."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

LessonId = Literal["epsilon-greedy", "linucb"]
Mode = Literal["guided", "challenge", "freePlay"]

GATE_IDS: Final[tuple[str, ...]] = ("moon", "sun", "star")
EPSILON_MEANS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
EPSILON_CHOICES: Final[tuple[float, ...]] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
ALPHA_CHOICES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
CUE_NAMES: Final[tuple[str, ...]] = ("light", "echo", "tide")
LINUCB_THETA: Final[NDArray[np.float64]] = np.array(
    [[0.1, -1.2, 0.2, -0.8], [0.0, 1.0, 0.3, 1.0], [0.2, 0.0, -1.1, 0.2]],
    dtype=float,
)


@dataclass(frozen=True)
class LessonFixture:
    """Immutable defaults and scoring targets for one lesson."""

    lesson_id: LessonId
    guided_seed: int
    challenge_seed: int
    guided_horizon: int
    challenge_horizon: int
    reward_target: int
    regret_target: float


FIXTURES: Final[dict[LessonId, LessonFixture]] = {
    "epsilon-greedy": LessonFixture("epsilon-greedy", 42, 7, 12, 20, 12, 3.25),
    "linucb": LessonFixture("linucb", 31415, 20260824, 12, 20, 10, 3.25),
}


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def validate_parameters(
    lesson_id: LessonId, parameters: dict[str, object]
) -> dict[str, float]:
    """Return normalized lesson parameters or raise a specific validation error."""

    if lesson_id == "epsilon-greedy":
        epsilon = _number(parameters.get("epsilon", 0.2), name="epsilon")
        if epsilon not in EPSILON_CHOICES:
            raise ValueError(f"epsilon must be one of {EPSILON_CHOICES}")
        return {"epsilon": epsilon}
    alpha = _number(parameters.get("alpha", 1.0), name="alpha")
    l2 = _number(parameters.get("l2", 1.0), name="l2")
    if alpha not in ALPHA_CHOICES:
        raise ValueError(f"alpha must be one of {ALPHA_CHOICES}")
    if l2 != 1.0:
        raise ValueError("l2 must be 1.0")
    return {"alpha": alpha, "l2": l2}


def horizon_for(lesson_id: LessonId, mode: Mode) -> int:
    """Resolve the fixed lesson horizon."""

    fixture = FIXTURES[lesson_id]
    return fixture.guided_horizon if mode == "guided" else fixture.challenge_horizon
