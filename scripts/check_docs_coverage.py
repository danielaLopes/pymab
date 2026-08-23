"""Enforce the total reported by Sphinx's Python coverage builder."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TOTAL_PATTERN = re.compile(
    r"^\|\s*TOTAL\s*\|\s*(?P<coverage>\d+(?:\.\d+)?)%\s*\|",
    re.MULTILINE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum", type=float, default=100.0)
    args = parser.parse_args()

    if not 0 <= args.minimum <= 100:
        parser.error("--minimum must be between 0 and 100")
    report = args.report.read_text(encoding="utf-8")
    match = TOTAL_PATTERN.search(report)
    if match is None:
        raise SystemExit(f"Could not find TOTAL coverage in {args.report}")
    coverage = float(match.group("coverage"))
    print(f"Sphinx API documentation coverage: {coverage:.2f}%")
    if coverage < args.minimum:
        raise SystemExit(
            f"API documentation coverage {coverage:.2f}% is below "
            f"the required {args.minimum:.2f}%"
        )


if __name__ == "__main__":
    main()
