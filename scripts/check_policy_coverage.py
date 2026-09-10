"""Verify complete Rust and Python coverage of every built-in policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pymab.policies as public_policies
from pymab import _native
from pymab._reference.registry import REFERENCE_POLICY_SPECS
from pymab.policies._native_mixin import NativePolicyMixin

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "policies"


def _rust_kinds() -> set[str]:
    source = (
        ROOT / "crates" / "pymab-core" / "src" / "policy" / "registry.rs"
    ).read_text(encoding="utf-8")
    section = source.split("pub const fn as_str", 1)[1].split(
        "pub const fn python_name", 1
    )[0]
    return set(re.findall(r'"([a-z][a-z0-9_]*)"', section))


def main() -> int:
    """Validate the complete built-in policy coverage matrix."""

    if not _native.native_available():
        raise SystemExit("native extension is required for the policy coverage gate")
    registry = json.loads((FIXTURES / "registry.json").read_text(encoding="utf-8"))
    entries = cast(list[dict[str, str]], registry["policies"])
    registered = {entry["rust_kind"] for entry in entries}
    fixture_paths = [
        path
        for path in FIXTURES.glob("*.json")
        if path.name not in {"registry.json", "schema.json"}
    ]
    fixtures = {
        str(payload["policy_kind"]): payload
        for path in fixture_paths
        for payload in [json.loads(path.read_text(encoding="utf-8"))]
    }
    sources = {
        "Rust registry": _rust_kinds(),
        "reference registry": set(REFERENCE_POLICY_SPECS),
        "shared fixtures": set(fixtures),
    }
    for label, kinds in sources.items():
        if kinds != registered:
            raise SystemExit(
                f"{label} differs from policy registry: "
                f"missing={sorted(registered - kinds)}, extra={sorted(kinds - registered)}"
            )
    for entry in entries:
        fixture = fixtures[entry["rust_kind"]]
        policy_type = cast(type[Any], getattr(public_policies, entry["python_name"]))
        policy = policy_type(**fixture["config"])
        if not isinstance(policy, NativePolicyMixin):
            raise SystemExit(f"{entry['python_name']} is not backed by a native handle")
        if policy._native_kind != entry["rust_kind"]:
            raise SystemExit(
                f"{entry['python_name']} maps to {policy._native_kind}, "
                f"expected {entry['rust_kind']}"
            )
        _native.create_policy(
            entry["rust_kind"],
            json.dumps(fixture["config"], sort_keys=True, separators=(",", ":")),
        )
    print(f"Policy coverage gate passed for {len(entries)} built-ins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
