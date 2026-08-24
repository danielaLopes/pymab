"""Single JSON entrypoint imported by CPython and Pyodide."""

from __future__ import annotations

import json
from typing import Any

import pymab
from pymab_demo.protocol import DemoError, dumps
from pymab_demo.sessions import LessonSession, create_session

_sessions: dict[str, LessonSession] = {}


def _error(request_id: str, error: DemoError) -> str:
    return dumps({"type": "error", "requestId": request_id, "error": error})


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an already decoded command and return a plain response mapping."""

    request_id = str(request.get("requestId", ""))
    command = request.get("type")
    if not request_id or not isinstance(command, str):
        raise ValueError("type and requestId are required")
    if command == "initialize":
        return {
            "type": "ready",
            "requestId": request_id,
            "packageVersion": pymab.__version__,
            "sourceCommit": str(request.get("sourceCommit", "unknown")),
        }
    session_id = str(request.get("sessionId", ""))
    if not session_id:
        raise ValueError("sessionId is required")
    if command == "startLesson":
        lesson_id = str(request.get("lessonId"))
        mode = str(request.get("mode"))
        if lesson_id not in ("epsilon-greedy", "linucb"):
            raise ValueError("unknown lessonId")
        if mode not in ("guided", "challenge", "freePlay"):
            raise ValueError("unknown mode")
        if session_id in _sessions and not _sessions[session_id].disposed:
            raise ValueError("sessionId is already active")
        new_session = create_session(
            session_id=session_id,
            lesson_id=lesson_id,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            seed=int(request.get("seed", 0)),
            parameters=dict(request.get("parameters", {})),
            source_commit=str(request.get("sourceCommit", "unknown")),
        )
        _sessions[session_id] = new_session
        return {
            "type": "lessonStarted",
            "requestId": request_id,
            "sessionId": session_id,
            "snapshot": new_session.snapshot(),
        }
    current_session = _sessions.get(session_id)
    if current_session is None or current_session.disposed:
        raise LookupError("session is missing or disposed")
    if command == "step":
        return {
            "type": "stepCompleted",
            "requestId": request_id,
            "sessionId": session_id,
            "snapshot": current_session.step(),
        }
    if command == "runToEnd":
        return {
            "type": "runCompleted",
            "requestId": request_id,
            "sessionId": session_id,
            "snapshot": current_session.run_to_end(),
        }
    if command == "reset":
        return {
            "type": "lessonStarted",
            "requestId": request_id,
            "sessionId": session_id,
            "snapshot": current_session.reset(),
        }
    if command == "dispose":
        current_session.dispose()
        return {"type": "disposed", "requestId": request_id, "sessionId": session_id}
    raise ValueError(f"unknown command: {command}")


def dispatch_json(request_json: str) -> str:
    """Decode, execute, and encode one protocol request without Pyodide imports."""

    request_id = ""
    try:
        request = json.loads(request_json)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        request_id = str(request.get("requestId", ""))
        return dumps(dispatch(request))
    except LookupError as exc:
        return _error(request_id, DemoError("INVALID_SESSION", str(exc), True))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _error(request_id, DemoError("INVALID_REQUEST", str(exc), True))
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        return _error(
            request_id, DemoError("POLICY_FAILED", str(exc), False, type(exc).__name__)
        )


def clear_sessions() -> None:
    """Clear process-global sessions for deterministic tests."""

    _sessions.clear()
