"""Canonical lesson fixtures shared by the browser and CPython tests."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from pymab_demo.catalog import POLICY_CATALOG, PolicyId

LessonId = PolicyId
Mode = Literal["guided", "challenge", "freePlay"]

GATE_IDS: Final[tuple[str, ...]] = ("moon", "sun", "star")
EPSILON_MEANS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
EPSILON_RANGE: Final[tuple[float, float, float]] = (0.0, 1.0, 0.01)
ALPHA_RANGE: Final[tuple[float, float, float]] = (0.05, 4.0, 0.05)
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


def _in_range_and_step(value: float, limits: tuple[float, float, float]) -> bool:
    minimum, maximum, step = limits
    steps = (value - minimum) / step
    return bool(minimum <= value <= maximum and np.isclose(steps, round(steps)))


def validate_parameters(
    lesson_id: LessonId, parameters: dict[str, object]
) -> dict[str, Any]:
    """Return normalized lesson parameters or raise a specific validation error."""

    if lesson_id == "epsilon-greedy":
        unknown = set(parameters) - {"epsilon", "initial_value"}
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
        epsilon = _number(parameters.get("epsilon", 0.2), name="epsilon")
        initial_value = _number(
            parameters.get("initial_value", 0.0), name="initial_value"
        )
        if not _in_range_and_step(epsilon, EPSILON_RANGE):
            raise ValueError("epsilon must be from 0 to 1 in steps of 0.01")
        if not np.isfinite(initial_value):
            raise ValueError("initial_value must be finite")
        return {"epsilon": epsilon, "initial_value": initial_value}
    if lesson_id == "linucb":
        alpha = _number(parameters.get("alpha", 1.0), name="alpha")
        l2 = _number(parameters.get("l2", 1.0), name="l2")
        if not _in_range_and_step(alpha, ALPHA_RANGE):
            raise ValueError("alpha must be from 0.05 to 4 in steps of 0.05")
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be positive")
        return {"alpha": alpha, "l2": l2}

    spec = POLICY_CATALOG[lesson_id]
    expected = set(spec.defaults)
    unknown = set(parameters) - expected
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
    normalized: dict[str, Any] = {}
    for name, default in spec.defaults.items():
        value = parameters.get(name, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            normalized[name] = value
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be text")
            normalized[name] = value
        elif default is None:
            normalized[name] = None if value is None else _number(value, name=name)
        else:
            normalized[name] = _number(value, name=name)
            if isinstance(default, int) and not isinstance(default, bool):
                if not float(normalized[name]).is_integer():
                    raise ValueError(f"{name} must be an integer")
                normalized[name] = int(normalized[name])
    spec.create(normalized, horizon=spec.horizon)
    return normalized


def validate_environment(
    lesson_id: LessonId, mode: Mode, environment: dict[str, object] | None
) -> dict[str, Any]:
    """Validate an optional free-play reward environment."""

    if environment is None:
        return {}
    if mode != "freePlay":
        raise ValueError("a custom environment is only available in free play")
    kind = POLICY_CATALOG[lesson_id].environment

    def vector(
        name: str, value: object, *, unit_interval: bool
    ) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must contain exactly three values")
        parsed = tuple(
            _number(item, name=f"{name}[{index}]") for index, item in enumerate(value)
        )
        values = (parsed[0], parsed[1], parsed[2])
        if any(not np.isfinite(item) for item in values):
            raise ValueError(f"{name} must contain finite values")
        if unit_interval and any(
            item < 0 or item > 1 or not np.isclose(item * 1000, round(item * 1000))
            for item in values
        ):
            raise ValueError(f"{name} must be from 0 to 1 in steps of 0.001")
        return values

    if kind in {"stationary-bernoulli", "best-arm"}:
        if set(environment) != {"probabilities"}:
            raise ValueError("environment must contain only probabilities")
        return {
            "probabilities": vector(
                "probabilities", environment["probabilities"], unit_interval=True
            )
        }
    if kind == "stationary-gaussian":
        if set(environment) != {"means", "standardDeviation"}:
            raise ValueError(
                "Gaussian environment requires means and standardDeviation"
            )
        standard_deviation = _number(
            environment["standardDeviation"], name="standardDeviation"
        )
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError("standardDeviation must be positive and finite")
        return {
            "means": vector("means", environment["means"], unit_interval=False),
            "standardDeviation": standard_deviation,
        }
    if kind == "changing-bernoulli":
        if set(environment) != {"phases"}:
            raise ValueError("changing environment requires phases")
        raw_phases = environment["phases"]
        if not isinstance(raw_phases, (list, tuple)) or not 2 <= len(raw_phases) <= 4:
            raise ValueError("phases must contain two to four entries")
        phases: list[dict[str, object]] = []
        starts: list[int] = []
        for index, phase in enumerate(raw_phases):
            if not isinstance(phase, dict) or set(phase) != {"start", "probabilities"}:
                raise ValueError("each phase requires start and probabilities")
            start = phase["start"]
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise ValueError("phase starts must be non-negative integers")
            starts.append(start)
            phases.append(
                {
                    "start": start,
                    "probabilities": vector(
                        f"phases[{index}].probabilities",
                        phase["probabilities"],
                        unit_interval=True,
                    ),
                }
            )
        if starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("phase starts must begin at zero and increase")
        return {"phases": phases}
    if kind == "adversarial":
        if set(environment) != {"rewards"}:
            raise ValueError("adversarial environment requires rewards")
        raw_rows = environment["rewards"]
        if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
            raise ValueError("rewards must contain at least one round")
        rows = [
            vector(f"rewards[{index}]", row, unit_interval=True)
            for index, row in enumerate(raw_rows)
        ]
        return {"rewards": rows}

    allowed = {"theta"}
    if kind == "contextual-linear":
        allowed.add("standardDeviation")
    if set(environment) != allowed:
        if "probabilities" in environment:
            raise ValueError(
                "custom probabilities are only available for non-contextual policies"
            )
        raise ValueError("contextual environment has unexpected fields")
    raw_theta = environment["theta"]
    if not isinstance(raw_theta, (list, tuple)) or len(raw_theta) != 3:
        raise ValueError("theta must contain three portal rows")
    theta: list[tuple[float, float, float, float]] = []
    for index, row in enumerate(raw_theta):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"theta[{index}] must contain four coefficients")
        parsed = tuple(
            _number(item, name=f"theta[{index}] coefficient") for item in row
        )
        if any(not np.isfinite(item) for item in parsed):
            raise ValueError("theta coefficients must be finite")
        theta.append((parsed[0], parsed[1], parsed[2], parsed[3]))
    result: dict[str, Any] = {"theta": theta}
    if kind == "contextual-linear":
        noise = _number(environment["standardDeviation"], name="standardDeviation")
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("standardDeviation must be positive and finite")
        result["standardDeviation"] = noise
    return result


def horizon_for(lesson_id: LessonId, mode: Mode) -> int:
    """Resolve the fixed lesson horizon."""

    fixture = FIXTURES.get(lesson_id)
    if fixture is not None:
        return fixture.guided_horizon if mode == "guided" else fixture.challenge_horizon
    spec = POLICY_CATALOG[lesson_id]
    return spec.horizon if mode == "guided" else spec.challenge_horizon
