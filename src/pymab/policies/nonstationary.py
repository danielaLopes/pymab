"""Compatibility imports for Python reference non-stationary policies."""

from pymab._reference.policies.nonstationary import (
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
)

__all__ = [
    "DiscountedBernoulliThompsonSamplingPolicy",
    "DiscountedUCBPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
]
