"""Compatibility imports for Python reference Thompson-sampling policies."""

from pymab._reference.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy as _BernoulliThompsonSamplingPolicy,
)
from pymab._reference.policies.thompson_sampling import (
    GaussianThompsonSamplingPolicy as _GaussianThompsonSamplingPolicy,
)
from pymab.policies._native_mixin import native_policy_class

BernoulliThompsonSamplingPolicy = native_policy_class(
    "bernoulli_thompson_sampling",
    _BernoulliThompsonSamplingPolicy,
    module=__name__,
)
GaussianThompsonSamplingPolicy = native_policy_class(
    "gaussian_thompson_sampling",
    _GaussianThompsonSamplingPolicy,
    module=__name__,
)

__all__ = ["BernoulliThompsonSamplingPolicy", "GaussianThompsonSamplingPolicy"]
