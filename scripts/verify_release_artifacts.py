"""Inspect the complete crate, sdist, and native-wheel release set."""

from __future__ import annotations

import argparse
import email
import tarfile
import tomllib
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).parents[1]
PYTHON_TAGS = ("cp311-cp311", "cp312-cp312", "cp313-cp313", "cp314-cp314")
PLATFORMS = {
    "linux-x86_64": ("manylinux", "x86_64"),
    "linux-aarch64": ("manylinux", "aarch64"),
    "macos-x86_64": ("macosx", "x86_64"),
    "macos-arm64": ("macosx", "arm64"),
    "windows-x86_64": ("win_amd64",),
}


def _workspace_version() -> str:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return str(cargo["workspace"]["package"]["version"])


def _wheel_platform(filename: str) -> str:
    matches = [
        label
        for label, markers in PLATFORMS.items()
        if all(marker in filename for marker in markers)
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot classify wheel platform: {filename}")
    return matches[0]


def verify(dist: Path, cargo_package: Path, *, expected_wheel_count: int = 20) -> None:
    """Raise on missing, duplicate, mismatched, or non-native artifacts."""

    version = _workspace_version()
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    crates = sorted(cargo_package.glob("*.crate"))
    if len(wheels) != expected_wheel_count:
        raise ValueError(f"expected {expected_wheel_count} wheels, found {len(wheels)}")
    if len(sdists) != 1 or len(crates) != 1:
        raise ValueError(
            f"expected one sdist and one crate, found {len(sdists)} and {len(crates)}"
        )
    expected_fragment = version.replace("-", "_")
    tags: Counter[str] = Counter()
    platforms: Counter[str] = Counter()
    for wheel in wheels:
        if expected_fragment not in wheel.name:
            raise ValueError(f"wheel version differs from {version}: {wheel.name}")
        tag_matches = [tag for tag in PYTHON_TAGS if tag in wheel.name]
        if len(tag_matches) != 1:
            raise ValueError(f"wheel has unexpected Python tag: {wheel.name}")
        tags[tag_matches[0]] += 1
        platforms[_wheel_platform(wheel.name)] += 1
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = email.message_from_bytes(archive.read(metadata_name))
            if metadata["Version"] != version:
                raise ValueError(f"wheel metadata version differs: {wheel.name}")
            if not any(
                "_pymab" in name and name.endswith((".so", ".pyd")) for name in names
            ):
                raise ValueError(f"wheel lacks native extension: {wheel.name}")
    if expected_wheel_count == 20 and (
        set(tags.values()) != {5} or set(platforms.values()) != {4}
    ):
        raise ValueError(f"incomplete wheel matrix: tags={tags}, platforms={platforms}")

    sdist = sdists[0]
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        required = ("pyproject.toml", "Cargo.toml", "crates/pymab-core/src/lib.rs")
        for suffix in required:
            if not any(name.endswith(suffix) for name in names):
                raise ValueError(f"sdist lacks {suffix}")

    crate = crates[0]
    if crate.name != f"pymab-{version}.crate":
        raise ValueError(f"unexpected crate filename: {crate.name}")
    with tarfile.open(crate, "r:gz") as archive:
        names = archive.getnames()
        for suffix in ("Cargo.toml", "README.md", "src/lib.rs"):
            if not any(name.endswith(suffix) for name in names):
                raise ValueError(f"crate lacks {suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--cargo-package", type=Path, required=True)
    parser.add_argument("--expected-wheel-count", type=int, default=20)
    arguments = parser.parse_args(argv)
    verify(
        arguments.dist,
        arguments.cargo_package,
        expected_wheel_count=arguments.expected_wheel_count,
    )
    print("Release artifact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
