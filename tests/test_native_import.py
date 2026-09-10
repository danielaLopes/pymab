from __future__ import annotations

import importlib

import pytest

from pymab import _native


def test_native_facade_is_safe_before_or_after_extension_build() -> None:
    available = _native.native_available()
    assert isinstance(available, bool)
    if not available:
        assert _native.core_version() is None
        assert _native.rng_scheme_version() is None


def test_compiled_extension_reports_core_metadata() -> None:
    if not _native.native_available():
        pytest.skip("native extension has not been built in this environment")

    extension = importlib.import_module("pymab._pymab")
    assert extension.native_available() is True
    assert extension.core_version() == _native.core_version()
    assert extension.rng_scheme_version() == _native.rng_scheme_version()
    assert extension.core_version()
    assert extension.rng_scheme_version().startswith("pymab-rust-")
