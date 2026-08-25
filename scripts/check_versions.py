"""Verify that dynamic Python metadata and Cargo workspace versions agree."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from pymab import _native

ROOT = Path(__file__).parents[1]


def declared_versions() -> dict[str, str]:
    """Collect every authoritative version declaration."""

    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    workspace = str(cargo["workspace"]["package"]["version"])
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    packages = {
        str(package["name"]): str(package["version"])
        for package in lock["package"]
        if package["name"] in {"pymab", "pymab-python"}
    }
    release_manifest = json.loads(
        (ROOT / ".release-please-manifest.json").read_text(encoding="utf-8")
    )
    return {
        "cargo-workspace": workspace,
        "cargo-lock-core": packages.get("pymab", "missing"),
        "cargo-lock-python": packages.get("pymab-python", "missing"),
        "python-metadata": importlib.metadata.version("pymab"),
        "release-manifest": str(release_manifest["."]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-native", action="store_true")
    arguments = parser.parse_args(argv)
    versions = declared_versions()
    expected = versions["cargo-workspace"]
    mismatches = {name: value for name, value in versions.items() if value != expected}
    if arguments.require_native:
        native = _native.core_version()
        if native is None:
            raise SystemExit("native extension is required for the version gate")
        if native != expected:
            mismatches["native-core"] = native
    if mismatches:
        details = ", ".join(f"{name}={value}" for name, value in mismatches.items())
        raise SystemExit(f"version mismatch; expected {expected}: {details}")
    print(f"Version gate passed: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
