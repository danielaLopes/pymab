"""Compatibility import for the Python reference adversarial policy."""

from pymab._reference.policies.adversarial import EXP3Policy as _EXP3Policy
from pymab.policies._native_mixin import native_policy_class

EXP3Policy = native_policy_class("exp3", _EXP3Policy, module=__name__)

__all__ = ["EXP3Policy"]
