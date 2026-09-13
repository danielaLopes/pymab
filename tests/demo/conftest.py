"""Test configuration for the browser-neutral demo bridge."""

from __future__ import annotations

import pytest
from pymab_demo.entrypoint import clear_sessions


@pytest.fixture(autouse=True)
def isolated_sessions() -> None:
    """Prevent process-global worker sessions leaking between tests."""

    clear_sessions()
