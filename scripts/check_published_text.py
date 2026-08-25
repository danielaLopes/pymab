"""Reject punctuation that does not match the website's plain-language style."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_TEXT = (
    ROOT / "README.md",
    ROOT / "website" / "index.html",
    ROOT / "web" / "index.html",
)
PUBLISHED_TREES = (
    (ROOT / "docs" / "source", {".md", ".py", ".rst"}),
    (ROOT / "web" / "src", {".ts", ".tsx"}),
)
FORBIDDEN = {
    "—": "em dash",
    "…": "Unicode ellipsis",
}


def published_files() -> list[Path]:
    """Return source files that can contribute user-facing website text."""
    files = list(PUBLISHED_TEXT)
    for directory, suffixes in PUBLISHED_TREES:
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    return sorted(files)


def main() -> int:
    """Print forbidden punctuation with source locations and return a status code."""
    failures: list[str] = []
    for path in published_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for character, name in FORBIDDEN.items():
                if character in line:
                    relative = path.relative_to(ROOT)
                    failures.append(
                        f"{relative}:{line_number}: found {name} ({character})"
                    )

    if failures:
        print("Published text contains forbidden punctuation:")
        print("\n".join(failures))
        return 1

    print("Published text uses the approved punctuation style.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
