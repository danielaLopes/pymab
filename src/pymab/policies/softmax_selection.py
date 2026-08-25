"""Compatibility import for the Python reference softmax policy."""

from pymab._reference.policies.softmax_selection import SoftmaxPolicy as _SoftmaxPolicy
from pymab.policies._native_mixin import native_policy_class

SoftmaxPolicy = native_policy_class("softmax", _SoftmaxPolicy, module=__name__)

__all__ = ["SoftmaxPolicy"]
