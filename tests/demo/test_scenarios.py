from __future__ import annotations

import json

import pytest
from pymab_demo.entrypoint import clear_sessions, dispatch
from pymab_demo.protocol import json_safe
from pymab_demo.scenarios import create_scenario_session

from pymab.policies import LinUCBPolicy, LogisticContextualBanditPolicy


def recommendation_environment(
    count: int, features: list[str] | None = None
) -> dict[str, object]:
    kinds = ("article", "product", "tutorial")
    feature_ids = ["visitor", "engagement", "visit"] if features is None else features
    return {
        "candidates": [
            {
                "id": f"candidate-{index + 1}",
                "name": f"Candidate {index + 1}",
                "symbolKind": kinds[index % len(kinds)],
            }
            for index in range(count)
        ],
        "features": feature_ids,
        "theta": [
            [0.05 * index, *[0.1 * (column + 1) for column in range(len(feature_ids))]]
            for index in range(count)
        ],
    }


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
    for arm in set(range(session.policy.n_arms)) - {selected}:
        assert all(before[arm] == after[arm])


@pytest.mark.parametrize("count", [2, 3, 8])
def test_recommendation_supports_dynamic_candidate_counts(count: int) -> None:
    session = create_scenario_session(
        session_id=f"recommendation-{count}",
        scenario_id="recommendations",
        mode="freePlay",
        seed=55,
        parameters={},
        source_commit="test",
        environment=recommendation_environment(count),
    )

    snapshot = session.run_to_end()
    assert session.policy.n_arms == count
    assert len(snapshot["presentation"]["arms"]) == count
    assert len(snapshot["gateIds"]) == count
    assert all(len(event["publicContext"]) == count for event in snapshot["history"])
    assert f"n_arms={count}" in snapshot["generatedCode"]

    replay = session.reset()
    assert [arm["name"] for arm in replay["presentation"]["arms"]] == [
        f"Candidate {index + 1}" for index in range(count)
    ]
    assert json_safe(session.run_to_end()["history"]) == json_safe(snapshot["history"])


def test_recommendation_preserves_custom_candidate_order_and_identity() -> None:
    environment = recommendation_environment(4)
    candidates = environment["candidates"]
    theta = environment["theta"]
    assert isinstance(candidates, list)
    assert isinstance(theta, list)
    environment["candidates"] = [candidates[3], candidates[1], candidates[0]]
    environment["theta"] = [theta[3], theta[1], theta[0]]
    session = create_scenario_session(
        session_id="recommendation-order",
        scenario_id="recommendations",
        mode="guided",
        seed=56,
        parameters={},
        source_commit="test",
        environment=environment,
    )

    assert [arm["id"] for arm in session.presentation()["arms"]] == [
        "candidate-4",
        "candidate-2",
        "candidate-1",
    ]
    assert session.snapshot()["gateIds"] == [
        "candidate-4",
        "candidate-2",
        "candidate-1",
    ]


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (recommendation_environment(1), "between 2 and 8"),
        (recommendation_environment(9), "between 2 and 8"),
        (
            {
                "candidates": [
                    {"id": "one", "name": "Same", "symbolKind": "article"},
                    {"id": "two", "name": " same ", "symbolKind": "product"},
                ],
                "theta": [[0.0] * 4, [0.0] * 4],
            },
            "names must be unique",
        ),
        (
            {
                "candidates": [
                    {"id": "one", "name": "One", "symbolKind": "article"},
                    {"id": "two", "name": "Two", "symbolKind": "avatar"},
                ],
                "theta": [[0.0] * 4, [0.0] * 4],
            },
            "symbolKind",
        ),
        (
            {
                "candidates": [
                    {"id": "one", "name": "One", "symbolKind": "article"},
                    {"id": "two", "name": "Two", "symbolKind": "product"},
                ],
                "theta": [[0.0] * 4],
            },
            "2 by 4",
        ),
    ],
)
def test_recommendation_rejects_invalid_candidate_configuration(
    environment: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        create_scenario_session(
            session_id="recommendation-invalid",
            scenario_id="recommendations",
            mode="freePlay",
            seed=57,
            parameters={},
            source_commit="test",
            environment=environment,
        )


@pytest.mark.parametrize(
    "symbol_kind",
    [
        "article",
        "product",
        "tutorial",
        "video",
        "podcast",
        "newsletter",
        "course",
        "event",
        "tool",
        "message",
        "offer",
        "download",
    ],
)
def test_recommendation_accepts_visual_types(symbol_kind: str) -> None:
    environment = recommendation_environment(2)
    candidates = environment["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["symbolKind"] = symbol_kind

    session = create_scenario_session(
        session_id=f"recommendation-{symbol_kind}",
        scenario_id="recommendations",
        mode="guided",
        seed=58,
        parameters={},
        source_commit="test",
        environment=environment,
    )

    assert session.presentation()["arms"][0]["symbolKind"] == symbol_kind


@pytest.mark.parametrize(
    "features",
    [
        [],
        ["device"],
        [
            "visitor",
            "engagement",
            "visit",
            "device",
            "account_age",
            "recent_activity",
            "price_sensitivity",
            "session_depth",
        ],
    ],
)
def test_recommendation_derives_features_and_context(features: list[str]) -> None:
    session = create_scenario_session(
        session_id=f"recommendation-features-{len(features)}",
        scenario_id="recommendations",
        mode="freePlay",
        seed=59,
        parameters={},
        source_commit="test",
        environment=recommendation_environment(3, features),
    )

    snapshot = session.step()
    event = snapshot["history"][0]
    assert session.policy.n_features == 1 + len(features)
    assert len(event["visibleCues"]) == len(features)
    assert all(len(row) == 1 + len(features) for row in event["publicContext"])
    assert all(-1 <= value <= 1 for value in event["publicContext"][0])
    assert [
        feature["id"] for feature in snapshot["presentation"]["contextFeatures"]
    ] == [
        "base",
        *features,
    ]
    assert f"n_features={1 + len(features)}" in snapshot["generatedCode"]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (["device", "device"], "feature ids must be unique"),
        (["unknown"], "feature id is not supported"),
        (["visitor"] * 9, "between 0 and 8"),
    ],
)
def test_recommendation_rejects_invalid_features(
    features: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_scenario_session(
            session_id="recommendation-invalid-features",
            scenario_id="recommendations",
            mode="freePlay",
            seed=60,
            parameters={},
            source_commit="test",
            environment=recommendation_environment(3, features),
        )


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
