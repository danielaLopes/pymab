from __future__ import annotations

import json

import pytest
from pymab_demo.protocol import dumps
from pymab_demo.sessions import create_session


def make_session(
    lesson: str = "epsilon-greedy",
    *,
    mode: str = "guided",
    seed: int = 42,
    parameters: dict[str, float] | None = None,
):
    return create_session(
        session_id="session",
        lesson_id=lesson,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        parameters=parameters
        or (
            {"epsilon": 0.2}
            if lesson == "epsilon-greedy"
            else {"alpha": 1.0, "l2": 1.0}
        ),
        source_commit="abc123",
    )


@pytest.mark.parametrize("lesson", ["epsilon-greedy", "linucb"])
def test_snapshots_are_json_safe_and_hide_truth_until_completion(lesson: str) -> None:
    session = make_session(lesson, seed=42 if lesson == "epsilon-greedy" else 31415)
    started = session.snapshot()
    assert started["hiddenTruth"] is None
    stepped = session.step()
    assert stepped["hiddenTruth"] is None
    assert json.loads(dumps(stepped))["step"] == 1
    completed = session.run_to_end()
    assert completed["hiddenTruth"] is not None
    assert completed["completed"] is True


@pytest.mark.parametrize("lesson", ["epsilon-greedy", "linucb"])
def test_reset_replays_every_public_value(lesson: str) -> None:
    session = make_session(lesson, seed=42 if lesson == "epsilon-greedy" else 31415)
    first = json.loads(dumps(session.run_to_end()))
    session.reset()
    replay = json.loads(dumps(session.run_to_end()))
    assert replay == first


@pytest.mark.parametrize(
    ("epsilon", "passed"), [(0.0, False), (0.2, True), (0.8, False)]
)
def test_epsilon_challenge_is_calibrated(epsilon: float, passed: bool) -> None:
    result = make_session(
        mode="challenge", seed=7, parameters={"epsilon": epsilon}
    ).run_to_end()
    assert result["passed"] is passed


@pytest.mark.parametrize(
    ("alpha", "passed"), [(0.25, False), (1.0, True), (2.0, False)]
)
def test_linucb_challenge_is_calibrated(alpha: float, passed: bool) -> None:
    result = make_session(
        "linucb",
        mode="challenge",
        seed=20260824,
        parameters={"alpha": alpha, "l2": 1.0},
    ).run_to_end()
    assert result["passed"] is passed


def test_disposed_session_rejects_steps() -> None:
    session = make_session()
    session.dispose()
    with pytest.raises(RuntimeError, match="disposed"):
        session.step()


def test_step_after_completion_is_an_idempotent_snapshot() -> None:
    session = make_session()
    completed = session.run_to_end()
    assert session.step() == completed


@pytest.mark.parametrize(
    ("lesson", "mode", "seed", "parameters", "actions", "rewards", "total", "regret"),
    [
        (
            "epsilon-greedy",
            "guided",
            42,
            {"epsilon": 0.2},
            [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            9,
            0.25,
        ),
        (
            "epsilon-greedy",
            "challenge",
            7,
            {"epsilon": 0.2},
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            12,
            3.0,
        ),
        (
            "linucb",
            "guided",
            31415,
            {"alpha": 1.0, "l2": 1.0},
            [1, 0, 0, 2, 2, 1, 2, 2, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            6,
            3.81301474,
        ),
        (
            "linucb",
            "challenge",
            20260824,
            {"alpha": 1.0, "l2": 1.0},
            [2, 1, 1, 0, 1, 1, 2, 0, 2, 1, 2, 1, 1, 0, 2, 1, 0, 2, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
            14,
            2.62676647,
        ),
    ],
)
def test_named_seed_trajectories_are_golden(
    lesson: str,
    mode: str,
    seed: int,
    parameters: dict[str, float],
    actions: list[int],
    rewards: list[int],
    total: int,
    regret: float,
) -> None:
    """Pin the exact teaching/challenge stories to the checked-out policy code."""

    result = make_session(
        lesson, mode=mode, seed=seed, parameters=parameters
    ).run_to_end()

    assert [event["selectedArm"] for event in result["history"]] == actions
    assert [event["reward"] for event in result["history"]] == rewards
    assert result["totalReward"] == total
    assert result["cumulativeExpectedRegret"] == pytest.approx(regret, abs=1e-8)
