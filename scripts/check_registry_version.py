"""Query whether an exact PyPI or crates.io package version already exists."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

Registry = Literal["crates", "pypi"]


def version_exists(registry: Registry, package: str, version: str) -> bool:
    """Return exact-version registry presence, propagating unexpected failures."""

    package_path = urllib.parse.quote(package, safe="")
    if registry == "pypi":
        request = urllib.request.Request(  # noqa: S310 - fixed HTTPS registry
            f"https://pypi.org/pypi/{package_path}/json",
            headers={
                "Accept": "application/json",
                "User-Agent": "pymab-release-check/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise
        return version in payload.get("releases", {})
    version_path = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS registry
        f"https://crates.io/api/v1/crates/{package_path}/{version_path}",
        headers={"Accept": "application/json", "User-Agent": "pymab-release-check/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20):  # noqa: S310
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", choices=("crates", "pypi"), required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    arguments = parser.parse_args(argv)
    deadline = time.monotonic() + arguments.wait_seconds
    exists = version_exists(
        cast(Registry, arguments.registry), arguments.package, arguments.version
    )
    while not exists and time.monotonic() < deadline:
        time.sleep(min(5, max(0, deadline - time.monotonic())))
        exists = version_exists(
            cast(Registry, arguments.registry), arguments.package, arguments.version
        )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"exists={str(exists).lower()}\n")
    print(
        f"{arguments.registry}:{arguments.package}:{arguments.version}: exists={exists}"
    )
    return 0 if exists or arguments.wait_seconds == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
