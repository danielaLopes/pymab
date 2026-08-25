"""Compatibility import for the Python reference random policy."""

from pymab._reference.policies.random import RandomPolicy as _RandomPolicy
from pymab.policies._native_mixin import native_policy_class

RandomPolicy = native_policy_class("random", _RandomPolicy, module=__name__)

__all__ = ["RandomPolicy"]
