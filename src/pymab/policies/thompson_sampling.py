"""Compatibility imports for Python reference Thompson-sampling policies."""

from pymab._reference.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
)

__all__ = ["BernoulliThompsonSamplingPolicy", "GaussianThompsonSamplingPolicy"]
