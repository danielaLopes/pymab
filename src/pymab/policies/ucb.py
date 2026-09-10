"""Compatibility imports for Python reference UCB policies."""

from pymab._reference.policies.ucb import (
    KLUCBPolicy as _KLUCBPolicy,
)
from pymab._reference.policies.ucb import (
    MOSSPolicy as _MOSSPolicy,
)
from pymab._reference.policies.ucb import (
    UCBPolicy as _UCBPolicy,
)
from pymab._reference.policies.ucb import (
    UCBStats,
)
from pymab.policies._native_mixin import native_policy_class

UCBPolicy = native_policy_class("ucb", _UCBPolicy, module=__name__)
KLUCBPolicy = native_policy_class("kl_ucb", _KLUCBPolicy, module=__name__)
MOSSPolicy = native_policy_class("moss", _MOSSPolicy, module=__name__)

__all__ = ["KLUCBPolicy", "MOSSPolicy", "UCBPolicy", "UCBStats"]
