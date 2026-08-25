"""Serve the assembled Pages artifact under its production ``/pymab/`` prefix."""

from __future__ import annotations

import argparse
import contextlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = REPOSITORY_ROOT / "website" / "build"
SITE_PREFIX = "/pymab"


class PagesPreviewHandler(SimpleHTTPRequestHandler):
    """Map the GitHub project prefix to a local artifact directory."""

    def do_GET(self) -> None:  # noqa: N802
        """Redirect the bare prefix and serve prefixed paths."""
        path = urlsplit(self.path).path
        if path == SITE_PREFIX:
            self.send_response(308)
            self.send_header("Location", f"{SITE_PREFIX}/")
            self.end_headers()
            return
        if not path.startswith(f"{SITE_PREFIX}/"):
            self.send_error(404, "Use /pymab/")
            return
        original_path = self.path
        self.path = self.path[len(SITE_PREFIX) :]
        try:
            super().do_GET()
        finally:
            self.path = original_path


def main() -> None:
    """Run the local production-path preview server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY, type=Path)
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    if not (directory / "index.html").is_file():
        raise SystemExit(
            f"Pages artifact is missing: {directory}. Run make pages-build first."
        )

    def handler(*handler_arguments: object, **handler_keywords: object) -> None:
        PagesPreviewHandler(
            *handler_arguments,
            directory=str(directory),
            **handler_keywords,
        )

    server = ThreadingHTTPServer((arguments.bind, arguments.port), handler)
    print(f"Serving PyMAB Pages at http://{arguments.bind}:{arguments.port}/pymab/")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
