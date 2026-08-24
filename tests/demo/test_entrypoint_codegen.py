from __future__ import annotations

import ast
import contextlib
import io
import json

import pymab_demo.entrypoint as entrypoint
import pytest
from pymab_demo.entrypoint import dispatch_json
from pymab_demo.protocol import json_safe
from pymab_demo.sessions import create_session


def send(**request: object) -> dict[str, object]:
    return json.loads(dispatch_json(json.dumps(request)))


def test_entrypoint_lifecycle_and_errors() -> None:
    ready = send(type="initialize", requestId="init", sourceCommit="abc")
    assert ready["type"] == "ready"
    started = send(
        type="startLesson",
        requestId="start",
        sessionId="s",
        lessonId="epsilon-greedy",
        mode="guided",
        seed=42,
        parameters={"epsilon": 0.2},
        sourceCommit="abc",
    )
    assert started["type"] == "lessonStarted"
    assert send(type="step", requestId="step", sessionId="s")["type"] == "stepCompleted"
    assert (
        send(type="reset", requestId="reset", sessionId="s")["type"] == "lessonStarted"
    )
    assert (
        send(type="runToEnd", requestId="run", sessionId="s")["type"] == "runCompleted"
    )
    assert (
        send(type="dispose", requestId="dispose", sessionId="s")["type"] == "disposed"
    )
    error = send(type="step", requestId="late", sessionId="s")
    assert error["type"] == "error"
    assert error["error"]["code"] == "INVALID_SESSION"  # type: ignore[index]


def test_active_duplicate_invalid_mode_and_unknown_command_are_rejected() -> None:
    payload = {
        "type": "startLesson",
        "requestId": "start",
        "sessionId": "s",
        "lessonId": "epsilon-greedy",
        "mode": "guided",
        "seed": 42,
        "parameters": {"epsilon": 0.2},
    }
    assert send(**payload)["type"] == "lessonStarted"
    duplicate = send(**{**payload, "requestId": "duplicate"})
    assert duplicate["error"]["code"] == "INVALID_REQUEST"  # type: ignore[index]
    invalid_mode = send(
        **{**payload, "requestId": "mode", "sessionId": "other", "mode": "unknown"}
    )
    assert invalid_mode["type"] == "error"
    unknown = send(type="unknown", requestId="unknown", sessionId="s")
    assert unknown["error"]["code"] == "INVALID_REQUEST"  # type: ignore[index]


def test_policy_failure_and_non_json_safe_values_are_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_request: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(entrypoint, "dispatch", fail)
    response = json.loads(
        entrypoint.dispatch_json('{"type":"initialize","requestId":"x"}')
    )
    assert response["error"]["code"] == "POLICY_FAILED"
    with pytest.raises(TypeError, match="not JSON-safe"):
        json_safe({1, 2, 3})


@pytest.mark.parametrize(
    "raw_request",
    [
        "not json",
        "[]",
        "{}",
        '{"type":"wat","requestId":"x","sessionId":"s"}',
        '{"type":"startLesson","requestId":"x","sessionId":"s","lessonId":"wat","mode":"guided"}',
    ],
)
def test_invalid_requests_are_structured(raw_request: str) -> None:
    response = json.loads(dispatch_json(raw_request))
    assert response["type"] == "error"
    assert response["error"]["code"] in {"INVALID_REQUEST", "INVALID_SESSION"}


@pytest.mark.parametrize(
    ("lesson", "seed", "parameters"),
    [
        ("epsilon-greedy", 42, {"epsilon": 0.2}),
        ("linucb", 31415, {"alpha": 1.0, "l2": 1.0}),
    ],
)
def test_generated_examples_parse_and_match_session_metrics(
    lesson: str, seed: int, parameters: dict[str, float]
) -> None:
    session = create_session(
        session_id="code",
        lesson_id=lesson,  # type: ignore[arg-type]
        mode="guided",
        seed=seed,
        parameters=parameters,
        source_commit="abc",
    )
    expected = session.run_to_end()
    code = session.generated_code()
    ast.parse(code)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<generated>", "exec"), {})  # noqa: S102
    actual = ast.literal_eval(output.getvalue().strip())
    assert actual["totalReward"] == expected["totalReward"]
    assert actual["cumulativeExpectedRegret"] == pytest.approx(
        expected["cumulativeExpectedRegret"]
    )
