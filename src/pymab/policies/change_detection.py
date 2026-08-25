"""Compatibility imports for Python reference change-detection policies."""

from pymab._reference.policies.change_detection import (
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
    PageHinkleyUCBPolicy,
)

__all__ = ["CUSUMUCBPolicy", "ChangePointUCBPolicy", "PageHinkleyUCBPolicy"]
