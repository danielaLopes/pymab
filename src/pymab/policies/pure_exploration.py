"""Compatibility imports for Python reference pure-exploration policies."""

from pymab._reference.policies.pure_exploration import (
    MedianEliminationPolicy as _MedianEliminationPolicy,
)
from pymab._reference.policies.pure_exploration import (
    SuccessiveEliminationPolicy as _SuccessiveEliminationPolicy,
)
from pymab.policies._native_mixin import native_policy_class

SuccessiveEliminationPolicy = native_policy_class(
    "successive_elimination", _SuccessiveEliminationPolicy, module=__name__
)
MedianEliminationPolicy = native_policy_class(
    "median_elimination", _MedianEliminationPolicy, module=__name__
)

__all__ = ["MedianEliminationPolicy", "SuccessiveEliminationPolicy"]
