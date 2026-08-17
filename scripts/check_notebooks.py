"""Execute checked-in notebook code cells without rewriting notebook files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def main() -> None:
    for path in sorted(Path("examples").glob("*.ipynb")):
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        namespace: dict[str, Any] = {"__name__": "__notebook__"}
        for cell_index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{path}:cell-{cell_index}", "exec"), namespace)
        print(f"executed {path}")


if __name__ == "__main__":
    main()
