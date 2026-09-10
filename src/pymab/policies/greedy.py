"""Compatibility import for the Python reference greedy policy."""

from pymab._reference.policies.greedy import GreedyPolicy as _GreedyPolicy
from pymab.policies._native_mixin import native_policy_class

GreedyPolicy = native_policy_class("greedy", _GreedyPolicy, module=__name__)

__all__ = ["GreedyPolicy"]
