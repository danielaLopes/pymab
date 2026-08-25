"""Assemble the PyMAB hub, Arcade, and documentation for GitHub Pages."""

from __future__ import annotations

import argparse
import html
import os
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_SOURCE = REPOSITORY_ROOT / "website"
DEFAULT_OUTPUT = WEBSITE_SOURCE / "build"
ARCADE_BUILD = REPOSITORY_ROOT / "web" / "dist"
DOCUMENTATION_BUILD = REPOSITORY_ROOT / "docs" / "build" / "html"
BRAND_ASSETS = (
    REPOSITORY_ROOT / "assets" / "pymab-mark.svg",
    REPOSITORY_ROOT / "assets" / "pymab-mark.png",
    REPOSITORY_ROOT / "assets" / "pymab-lucky-lever.png",
)


def require_file(path: Path) -> None:
    """Fail with a useful message when a required input is unavailable."""
    if not path.is_file():
        raise SystemExit(f"Missing required Pages input: {path}")


def require_directory(path: Path) -> None:
    """Fail with a useful message when a required input directory is unavailable."""
    if not path.is_dir():
        raise SystemExit(f"Missing required Pages input: {path}")


def redirect_document(target: str) -> str:
    """Return a small redirect page that preserves query strings and fragments."""
    escaped_target = html.escape(target, quote=True)
    javascript_target = target.replace("\\", "\\\\").replace('"', '\\"')
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="0; url={escaped_target}">
    <link rel="canonical" href="{escaped_target}">
    <title>Documentation moved · PyMAB</title>
  </head>
  <body>
    <p>This documentation page moved to <a href="{escaped_target}">{escaped_target}</a>.</p>
    <script>
      const target = new URL("{javascript_target}", window.location.href);
      target.search = window.location.search;
      target.hash = window.location.hash;
      window.location.replace(target);
    </script>
  </body>
</html>
"""


def create_legacy_redirects(output: Path) -> int:
    """Redirect every former root documentation HTML page into ``docs/``."""
    count = 0
    for documentation_page in sorted((output / "docs").rglob("*.html")):
        relative_page = documentation_page.relative_to(output / "docs")
        if relative_page == Path("index.html"):
            continue
        redirect_path = output / relative_page
        redirect_path.parent.mkdir(parents=True, exist_ok=True)
        target_path = Path("docs") / relative_page
        target = os.path.relpath(target_path, start=relative_page.parent).replace(
            os.sep, "/"
        )
        redirect_path.write_text(redirect_document(target), encoding="utf-8")
        count += 1
    return count


def validate_site(output: Path) -> None:
    """Validate the assembled route contract and critical asset references."""
    required_files = (
        output / "index.html",
        output / "styles.css",
        output / "assets" / "pymab-mark.svg",
        output / "assets" / "pymab-lucky-lever.png",
        output / "demo" / "index.html",
        output / "demo" / "pymab-mark.svg",
        output / "docs" / "index.html",
        output / "docs" / "_static" / "pymab-mark.svg",
        output / "policies.html",
        output / ".nojekyll",
    )
    for required_file in required_files:
        require_file(required_file)

    hub_html = (output / "index.html").read_text(encoding="utf-8")
    for expected_link in ('href="demo/"', 'href="docs/"'):
        if expected_link not in hub_html:
            raise SystemExit(f"Hub is missing required route: {expected_link}")

    arcade_html = (output / "demo" / "index.html").read_text(encoding="utf-8")
    if 'href="/pymab/demo/pymab-mark.svg"' not in arcade_html:
        raise SystemExit("Arcade was not built with the /pymab/demo/ base path")

    legacy_redirect = (output / "policies.html").read_text(encoding="utf-8")
    if "docs/policies.html" not in legacy_redirect:
        raise SystemExit("Legacy documentation redirect does not target /docs/")


def build_site(output: Path) -> None:
    """Assemble and validate a complete Pages artifact."""
    require_file(WEBSITE_SOURCE / "index.html")
    require_file(WEBSITE_SOURCE / "styles.css")
    require_directory(ARCADE_BUILD)
    require_directory(DOCUMENTATION_BUILD)
    for asset in BRAND_ASSETS:
        require_file(asset)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    shutil.copy2(WEBSITE_SOURCE / "index.html", output / "index.html")
    shutil.copy2(WEBSITE_SOURCE / "styles.css", output / "styles.css")
    assets_output = output / "assets"
    assets_output.mkdir()
    for asset in BRAND_ASSETS:
        shutil.copy2(asset, assets_output / asset.name)

    shutil.copytree(ARCADE_BUILD, output / "demo")
    shutil.copytree(DOCUMENTATION_BUILD, output / "docs")
    (output / ".nojekyll").touch()
    redirect_count = create_legacy_redirects(output)
    validate_site(output)
    print(
        f"Assembled PyMAB Pages site at {output} with {redirect_count} legacy redirects"
    )


def main() -> None:
    """Parse CLI options and build the site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    build_site(arguments.output.resolve())


if __name__ == "__main__":
    main()
