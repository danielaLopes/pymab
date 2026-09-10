from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "policies"
REGISTRY_FIELDS = frozenset({"schema_version", "policies"})
REGISTRY_ENTRY_FIELDS = frozenset({"python_name", "rust_kind"})
POLICY_FIXTURE_FIELDS = frozenset(
    {
        "schema_version",
        "policy_kind",
        "config",
        "updates",
        "checkpoints",
        "recommendation",
        "reset_state",
        "expected_error",
    }
)


def _strict_fields(
    payload: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, unknown={unknown}")


def load_registry(path: Path | None = None) -> tuple[Mapping[str, str], ...]:
    """Load the strict cross-language policy registry."""

    registry_path = path or FIXTURE_ROOT / "registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("registry must be an object")
    _strict_fields(payload, REGISTRY_FIELDS, name="registry")
    if payload["schema_version"] != 1 or not isinstance(payload["policies"], list):
        raise ValueError("registry schema_version or policies is invalid")

    entries: list[Mapping[str, str]] = []
    for index, value in enumerate(payload["policies"]):
        if not isinstance(value, dict):
            raise ValueError(f"registry.policies[{index}] must be an object")
        _strict_fields(value, REGISTRY_ENTRY_FIELDS, name=f"registry.policies[{index}]")
        if any(
            not isinstance(value[field], str) or not value[field] for field in value
        ):
            raise ValueError(
                f"registry.policies[{index}] values must be non-empty strings"
            )
        entries.append(cast(Mapping[str, str], value))
    return tuple(entries)


def validate_policy_fixture(payload: object) -> Mapping[str, object]:
    """Validate the shared top-level policy-fixture contract."""

    if not isinstance(payload, dict):
        raise ValueError("policy fixture must be an object")
    _strict_fields(payload, POLICY_FIXTURE_FIELDS, name="policy fixture")
    if payload["schema_version"] != 1:
        raise ValueError("policy fixture schema_version must be one")
    if not isinstance(payload["policy_kind"], str) or not payload["policy_kind"]:
        raise ValueError("policy_kind must be a non-empty string")
    if not isinstance(payload["config"], dict):
        raise ValueError("config must be an object")
    if not isinstance(payload["updates"], list):
        raise ValueError("updates must be a list")
    if not isinstance(payload["checkpoints"], list):
        raise ValueError("checkpoints must be a list")
    expected_error = payload["expected_error"]
    if expected_error is not None and not isinstance(expected_error, str):
        raise ValueError("expected_error must be null or a string")
    return cast(Mapping[str, object], payload)


def load_policy_fixture(path: Path) -> Mapping[str, object]:
    """Load one strict policy fixture."""

    return validate_policy_fixture(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def policy_registry() -> tuple[Mapping[str, str], ...]:
    """Return the strict shared policy registry."""

    return load_registry()
