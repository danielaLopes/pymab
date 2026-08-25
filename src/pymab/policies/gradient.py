"""Compatibility import for the Python reference gradient policy."""

from pymab._reference.policies.gradient import (
    GradientBanditPolicy as _GradientBanditPolicy,
)
from pymab.policies._native_mixin import native_policy_class

GradientBanditPolicy = native_policy_class(
    "gradient_bandit", _GradientBanditPolicy, module=__name__
)

__all__ = ["GradientBanditPolicy"]
