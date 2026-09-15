"""Build the pure-Python PyMAB wheel used by the Pyodide runtime."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _record_hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _record(entries: dict[str, bytes], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, content in sorted(entries.items()):
        writer.writerow((name, _record_hash(content), len(content)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def build(repository: Path, destination: Path) -> Path:
    """Package checked-out Python sources without the native extension."""

    cargo = (repository / "Cargo.toml").read_text(encoding="utf-8")
    workspace_package = cargo.partition("[workspace.package]")[2].partition("[")[0]
    version_match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*(?:#.*)?$',
        workspace_package,
        re.MULTILINE,
    )
    if version_match is None:
        raise ValueError("Cargo.toml does not define workspace.package.version")
    version = version_match.group(1)
    distribution = f"pymab-{version}"
    dist_info = f"{distribution}.dist-info"
    source = repository / "src" / "pymab"
    entries = {
        path.relative_to(source.parent).as_posix(): path.read_bytes()
        for path in sorted(source.rglob("*.py"))
        if path.is_file()
    }
    entries[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.4\n"
        "Name: pymab\n"
        f"Version: {version}\n"
        "Summary: Reliable, statistically rigorous multi-armed bandit experiments.\n"
        "Requires-Python: >=3.11\n"
        "Requires-Dist: numpy>=1.24\n"
        "\n"
    ).encode()
    entries[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: pymab-web\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
        b"\n"
    )
    record_name = f"{dist_info}/RECORD"
    entries[record_name] = _record(entries, record_name)

    destination.mkdir(parents=True, exist_ok=True)
    wheel = destination / f"{distribution}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(_archive_info(name), content)
    return wheel


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: package_pure_python_wheel.py REPOSITORY DESTINATION")
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
