"""Typed, JSON-safe response helpers for the worker boundary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class DemoError:
    """Structured error sent to the browser."""

    code: str
    message: str
    recoverable: bool
    details: str | None = None


def json_safe(value: Any) -> JsonValue:
    """Recursively convert dataclasses and NumPy values to plain JSON values."""

    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def dumps(value: object) -> str:
    """Serialize a protocol value deterministically."""

    return json.dumps(json_safe(value), separators=(",", ":"), sort_keys=True)
