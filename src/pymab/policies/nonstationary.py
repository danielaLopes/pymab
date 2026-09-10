"""Compatibility imports for Python reference non-stationary policies."""

from pymab._reference.policies.nonstationary import (
    DiscountedBernoulliThompsonSamplingPolicy as _DiscountedBernoulliThompsonSamplingPolicy,
)
from pymab._reference.policies.nonstationary import (
    DiscountedUCBPolicy as _DiscountedUCBPolicy,
)
from pymab._reference.policies.nonstationary import (
    SlidingWindowBernoulliThompsonSamplingPolicy as _SlidingWindowBernoulliThompsonSamplingPolicy,
)
from pymab._reference.policies.nonstationary import (
    SlidingWindowUCBPolicy as _SlidingWindowUCBPolicy,
)
from pymab.policies._native_mixin import native_policy_class

SlidingWindowUCBPolicy = native_policy_class(
    "sliding_window_ucb", _SlidingWindowUCBPolicy, module=__name__
)
DiscountedUCBPolicy = native_policy_class(
    "discounted_ucb", _DiscountedUCBPolicy, module=__name__
)
SlidingWindowBernoulliThompsonSamplingPolicy = native_policy_class(
    "sliding_window_bernoulli_thompson_sampling",
    _SlidingWindowBernoulliThompsonSamplingPolicy,
    module=__name__,
)
DiscountedBernoulliThompsonSamplingPolicy = native_policy_class(
    "discounted_bernoulli_thompson_sampling",
    _DiscountedBernoulliThompsonSamplingPolicy,
    module=__name__,
)

__all__ = [
    "DiscountedBernoulliThompsonSamplingPolicy",
    "DiscountedUCBPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
]
