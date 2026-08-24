from __future__ import annotations

from collections.abc import Mapping

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
