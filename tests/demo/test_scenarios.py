from __future__ import annotations

import json

import pytest
from pymab_demo.entrypoint import clear_sessions, dispatch
from pymab_demo.protocol import json_safe
from pymab_demo.scenarios import create_scenario_session

from pymab.policies import LinUCBPolicy, LogisticContextualBanditPolicy


@pytest.mark.parametrize(
    ("scenario_id", "policy_type", "policy_id"),
    [
        (
            "recommendations",
            LogisticContextualBanditPolicy,
            "logistic-contextual-bandit",
        ),
        ("defensive-verification", LinUCBPolicy, "linucb"),
    ],
)
def test_scenarios_use_public_pymab_policies(
    scenario_id: str, policy_type: type[object], policy_id: str
) -> None:
    session = create_scenario_session(
        session_id="scenario",
        scenario_id=scenario_id,
        mode="guided",
        seed=12,
        parameters={},
        source_commit="test",
    )

    assert isinstance(session.policy, policy_type)
    snapshot = session.step()
    assert snapshot["scenarioId"] == scenario_id
    assert snapshot["policyId"] == policy_id
    assert snapshot["presentation"]["experienceKind"] == "scenario"
    json.dumps(json_safe(snapshot))


@pytest.mark.parametrize("scenario_id", ["recommendations", "defensive-verification"])
def test_scenario_reset_replays_the_same_run(scenario_id: str) -> None:
    session = create_scenario_session(
        session_id="scenario",
        scenario_id=scenario_id,
        mode="challenge",
        seed=31,
        parameters={},
        source_commit="test",
    )
    first = session.run_to_end()
    replay = session.reset()
    assert replay["step"] == 0
    second = session.run_to_end()

    assert json_safe(first["history"]) == json_safe(second["history"])
    assert first["totalReward"] == second["totalReward"]
    assert first["cumulativeExpectedRegret"] == second["cumulativeExpectedRegret"]


def test_recommendation_feedback_is_binary_and_updates_only_selected_arm() -> None:
    session = create_scenario_session(
        session_id="recommendation",
        scenario_id="recommendations",
        mode="guided",
        seed=44,
        parameters={},
        source_commit="test",
    )
    snapshot = session.step()
    event = snapshot["history"][0]
    before = event["diagnostic"]["thetaBefore"]
    after = event["diagnostic"]["thetaAfter"]
    selected = event["selectedArm"]

    assert event["reward"] in (0, 1)
    assert "trueProbabilities" not in event["diagnostic"]
    assert any(before[selected] != after[selected])
    for arm in {0, 1, 2} - {selected}:
        assert all(before[arm] == after[arm])


def test_defensive_expected_regret_matches_the_simulated_utility_model() -> None:
    session = create_scenario_session(
        session_id="defense",
        scenario_id="defensive-verification",
        mode="guided",
        seed=45,
        parameters={},
        source_commit="test",
    )
    event = session.step()["history"][0]
    expected = session.truth_history[-1]["expectedUtilities"]
    chosen = event["selectedArm"]

    assert event["instantaneousExpectedRegret"] == pytest.approx(
        max(expected) - expected[chosen]
    )
    assert event["diagnostic"]["outcomeLabel"]
    assert "abuseProbability" not in event["diagnostic"]
    assert "expectedRewards" not in event["diagnostic"]


def test_entrypoint_starts_and_disposes_scenario_sessions() -> None:
    clear_sessions()
    started = dispatch(
        {
            "type": "startScenario",
            "requestId": "r1",
            "sessionId": "s1",
            "scenarioId": "recommendations",
            "mode": "guided",
            "seed": 1,
            "parameters": {},
        }
    )
    assert started["type"] == "lessonStarted"
    assert started["snapshot"]["scenarioId"] == "recommendations"
    disposed = dispatch({"type": "dispose", "requestId": "r2", "sessionId": "s1"})
    assert disposed["type"] == "disposed"
