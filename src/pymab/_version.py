"""Installed package version resolution."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymab")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "2.0.0.dev0"

__all__: list[str] = []
