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
RELEASE_VERSION_MARKER = "# x-release-please-version"
REQUIRED_RELEASE_EXTRA_FILES = (
    {"type": "generic", "path": "Cargo.toml"},
    {
        "type": "toml",
        "path": "Cargo.lock",
        "jsonpath": "$['package'][?(@.name.value=='pymab')].version",
    },
    {
        "type": "toml",
        "path": "Cargo.lock",
        "jsonpath": "$['package'][?(@.name.value=='pymab-python')].version",
    },
    {
        "type": "toml",
        "path": "uv.lock",
        "jsonpath": "$['package'][?(@.name.value=='pymab')].version",
    },
)
WORKSPACE_MEMBER_MANIFESTS = (
    Path("crates/pymab-core/Cargo.toml"),
    Path("crates/pymab-python/Cargo.toml"),
)
RELEASE_WORKFLOW = Path(".github/workflows/release-please.yml")
PUBLISH_WORKFLOW = Path(".github/workflows/release.yml")
REQUIRED_RELEASE_CLI_FRAGMENTS = (
    "release-please@17.6.0 github-release",
    'release-please@17.6.0 "${args[@]}"',
    "--config-file=release-please-config.json",
    "--manifest-file=.release-please-manifest.json",
    '--release-as="${RELEASE_AS}"',
    '"repos/${GITHUB_REPOSITORY}/git/ref/tags/v${workspace_version}"',
)
REQUIRED_PUBLISH_WORKFLOW_FRAGMENTS = (
    "types: [published]",
    "cargo publish -p pymab --locked --token",
    "name: crates-io",
    "name: pypi",
    "needs: [metadata, verify, registry-state, publish-crate]",
)


def release_configuration_errors() -> list[str]:
    """Return problems that would break synchronized release version updates."""

    errors: list[str] = []
    config = json.loads(
        (ROOT / "release-please-config.json").read_text(encoding="utf-8")
    )
    if config.get("release-type") != "simple":
        errors.append("release-type must be simple")

    packages = config.get("packages")
    root_package = packages.get(".") if isinstance(packages, dict) else None
    extra_files = (
        root_package.get("extra-files") if isinstance(root_package, dict) else None
    )
    if not isinstance(extra_files, list):
        errors.append("packages['.'].extra-files must be a list")
    else:
        for required in REQUIRED_RELEASE_EXTRA_FILES:
            if required not in extra_files:
                errors.append(f"missing Release Please extra-file entry: {required}")

    cargo_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    marked_version_lines = [
        line.strip()
        for line in cargo_text.splitlines()
        if RELEASE_VERSION_MARKER in line
    ]
    if len(marked_version_lines) != 1 or not marked_version_lines[0].startswith(
        "version = "
    ):
        errors.append(
            "Cargo.toml must mark exactly one workspace version with "
            f"{RELEASE_VERSION_MARKER}"
        )

    for relative_path in WORKSPACE_MEMBER_MANIFESTS:
        manifest = tomllib.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        if manifest.get("package", {}).get("version") != {"workspace": True}:
            errors.append(f"{relative_path} must inherit version.workspace")

    bindings_manifest = tomllib.loads(
        (ROOT / "crates/pymab-python/Cargo.toml").read_text(encoding="utf-8")
    )
    if bindings_manifest.get("package", {}).get("publish") is not False:
        errors.append("crates/pymab-python/Cargo.toml must set publish = false")

    release_workflow = (ROOT / RELEASE_WORKFLOW).read_text(encoding="utf-8")
    if "googleapis/release-please-action@" in release_workflow:
        errors.append(
            f"{RELEASE_WORKFLOW} must use the CLI while the action drops "
            "release-as in manifest mode"
        )
    for fragment in REQUIRED_RELEASE_CLI_FRAGMENTS:
        if fragment not in release_workflow:
            errors.append(
                f"{RELEASE_WORKFLOW} is missing required CLI argument: {fragment}"
            )
    github_release_position = release_workflow.find(
        "release-please@17.6.0 github-release"
    )
    release_pr_position = release_workflow.find('release-please@17.6.0 "${args[@]}"')
    if (
        github_release_position >= 0
        and release_pr_position >= 0
        and github_release_position > release_pr_position
    ):
        errors.append(f"{RELEASE_WORKFLOW} must run github-release before release-pr")

    publish_workflow = (ROOT / PUBLISH_WORKFLOW).read_text(encoding="utf-8")
    for fragment in REQUIRED_PUBLISH_WORKFLOW_FRAGMENTS:
        if fragment not in publish_workflow:
            errors.append(
                f"{PUBLISH_WORKFLOW} is missing required publication step: {fragment}"
            )

    return errors


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
    configuration_errors = release_configuration_errors()
    if configuration_errors:
        raise SystemExit(
            "release configuration mismatch: " + "; ".join(configuration_errors)
        )
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
