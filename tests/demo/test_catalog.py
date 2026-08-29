from __future__ import annotations

import ast
import contextlib
import io

import pytest
from pymab_demo.catalog import (
    POLICY_CATALOG,
    PolicySpec,
    catalog_payload,
    public_policy_classes,
)
from pymab_demo.protocol import json_safe
from pymab_demo.sessions import create_session

from pymab import policies
from pymab.policies.policy import ContextualPolicy, Policy


def free_play_environment(spec: PolicySpec, horizon: int) -> dict[str, object]:
    kind = spec.environment
    if kind in {"stationary-bernoulli", "best-arm"}:
        return {"probabilities": [0.2, 0.5, 0.8]}
    if kind == "stationary-gaussian":
        return {"means": [-0.2, 0.3, 0.9], "standardDeviation": 0.4}
    if kind == "changing-bernoulli":
        return {
            "phases": [
                {"start": 0, "probabilities": [0.8, 0.5, 0.2]},
                {"start": horizon // 2, "probabilities": [0.2, 0.5, 0.8]},
            ]
        }
    if kind == "adversarial":
        return {
            "rewards": [
                [1.0 if arm == round_index % 3 else 0.1 for arm in range(3)]
                for round_index in range(horizon)
            ]
        }
    environment: dict[str, object] = {
        "theta": [
            [0.1, -1.2, 0.2, -0.8],
            [0.0, 1.0, 0.3, 1.0],
            [0.2, 0.0, -1.1, 0.2],
        ]
    }
    if kind == "contextual-linear":
        environment["standardDeviation"] = 0.2
    return environment


def test_catalog_covers_every_concrete_public_policy_exactly_once() -> None:
    abstract = {"Policy", "ActionValuePolicy", "ContextualPolicy"}
    public = set(policies.__all__) - abstract
    assert len(POLICY_CATALOG) == 27
    assert public_policy_classes() == public
    assert len(public_policy_classes()) == len(POLICY_CATALOG)


def test_every_catalog_default_constructs_the_public_class() -> None:
    for spec in POLICY_CATALOG.values():
        policy = spec.create(spec.defaults, horizon=spec.horizon)
        assert type(policy) is spec.policy_class
        assert isinstance(policy, (Policy, ContextualPolicy))


def test_catalog_payload_is_complete_and_json_safe() -> None:
    payload = catalog_payload()
    assert json_safe(payload) == payload
    assert {item["policyId"] for item in payload} == set(POLICY_CATALOG)
    for item in payload:
        assert item["className"]
        assert item["family"]
        assert item["environment"]
        assert item["objective"]


def test_every_additional_policy_steps_resets_and_completes() -> None:
    for spec in POLICY_CATALOG.values():
        if spec.policy_id in {"epsilon-greedy", "linucb"}:
            continue
        session = create_session(
            session_id=f"session-{spec.policy_id}",
            lesson_id=spec.policy_id,
            mode="guided",
            seed=spec.seed,
            parameters=spec.guided,
            source_commit="catalog-test",
        )
        started = session.snapshot()
        assert started["step"] == 0
        assert started["hiddenTruth"] is None
        first = session.step()
        assert first["step"] == 1
        assert first["diagnostic"]["after"]["policyClass"] == spec.policy_class.__name__
        reset = session.reset()
        assert reset["step"] == 0
        completed = session.run_to_end()
        assert completed["completed"] is True
        assert completed["hiddenTruth"] is not None


def test_every_catalog_example_replays_final_metrics() -> None:
    for spec in POLICY_CATALOG.values():
        session = create_session(
            session_id=f"code-{spec.policy_id}",
            lesson_id=spec.policy_id,
            mode="guided",
            seed=spec.seed,
            parameters=spec.guided,
            source_commit="catalog-code-test",
        )
        expected = session.run_to_end()
        code = session.generated_code()
        ast.parse(code)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"<{spec.policy_id}>", "exec"), {})  # noqa: S102
        actual = ast.literal_eval(output.getvalue().strip())
        assert actual["totalReward"] == pytest.approx(expected["totalReward"])
        assert actual["cumulativeExpectedRegret"] == pytest.approx(
            expected["cumulativeExpectedRegret"]
        )
        if spec.objective == "best-arm":
            assert actual["recommendation"] == expected["recommendation"]


@pytest.mark.parametrize("mode", ["guided", "challenge", "freePlay"])
def test_every_policy_completes_every_mode_deterministically(mode: str) -> None:
    for spec in POLICY_CATALOG.values():
        horizon = spec.horizon if mode == "guided" else spec.challenge_horizon
        environment = (
            free_play_environment(spec, horizon) if mode == "freePlay" else None
        )
        kwargs = {
            "lesson_id": spec.policy_id,
            "mode": mode,
            "seed": spec.seed,
            "parameters": spec.guided if mode != "freePlay" else spec.defaults,
            "source_commit": "determinism-test",
            "environment": environment,
        }
        first = create_session(session_id=f"first-{spec.policy_id}", **kwargs)  # type: ignore[arg-type]
        replay = create_session(session_id=f"replay-{spec.policy_id}", **kwargs)  # type: ignore[arg-type]
        first_result = first.run_to_end()
        replay_result = replay.run_to_end()
        assert json_safe(first_result["history"]) == json_safe(replay_result["history"])
        assert first_result["totalReward"] == replay_result["totalReward"]
        assert (
            first_result["cumulativeExpectedRegret"]
            == replay_result["cumulativeExpectedRegret"]
        )
