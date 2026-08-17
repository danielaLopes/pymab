"""Execute Python fenced-code snippets from a Markdown document in order."""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import textwrap
from pathlib import Path

PYTHON_FENCE = re.compile(
    r"^```python[^\n]*\n(?P<code>.*?)^```\s*$", re.MULTILINE | re.DOTALL
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    args = parser.parse_args()

    source = args.document.read_text(encoding="utf-8")
    snippets = [
        textwrap.dedent(match.group("code")) for match in PYTHON_FENCE.finditer(source)
    ]
    if not snippets:
        raise SystemExit(f"No Python snippets found in {args.document}")

    namespace: dict[str, object] = {"__name__": "__documentation__"}
    for index, snippet in enumerate(snippets, start=1):
        code = compile(snippet, f"{args.document}:python-snippet-{index}", "exec")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    print(f"Executed {len(snippets)} Python snippets from {args.document}")


if __name__ == "__main__":
    main()
