"""Create a byte-for-byte deterministic bridge zip."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main(source: Path, destination: Path) -> None:
    """Archive Python sources with fixed metadata and sorted entries."""

    files = sorted(path for path in source.rglob("*.py") if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            relative = file.relative_to(source.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, file.read_bytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: package_bridge.py SOURCE DESTINATION")
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
