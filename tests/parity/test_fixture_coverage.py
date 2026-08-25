from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pymab.policies as policies

ABSTRACT_POLICY_NAMES = {"ActionValuePolicy", "ContextualPolicy", "Policy"}


def test_fixture_registry_covers_all_concrete_python_exports(
    policy_registry: tuple[Mapping[str, str], ...],
) -> None:
    exported = set(policies.__all__) - ABSTRACT_POLICY_NAMES
    registered = {entry["python_name"] for entry in policy_registry}
    rust_kinds = [entry["rust_kind"] for entry in policy_registry]

    assert registered == exported
    assert len(policy_registry) == 27
    assert len(rust_kinds) == len(set(rust_kinds))


def test_each_registered_policy_has_exactly_one_parity_fixture(
    policy_registry: tuple[Mapping[str, str], ...],
) -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures" / "policies"
    fixture_kinds: list[str] = []
    for path in sorted(fixture_root.glob("*.json")):
        if path.name in {"registry.json", "schema.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        kind = payload.get("policy_kind")
        assert isinstance(kind, str), f"{path.name} must declare policy_kind"
        fixture_kinds.append(kind)

    registered = sorted(entry["rust_kind"] for entry in policy_registry)
    assert sorted(fixture_kinds) == registered
