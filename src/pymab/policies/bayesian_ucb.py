"""Compatibility imports for the Python reference Bayesian-UCB policies."""

from pymab._reference.policies.bayesian_ucb import (
    BernoulliBayesianUCBPolicy as _BernoulliBayesianUCBPolicy,
)
from pymab._reference.policies.bayesian_ucb import (
    GaussianBayesianUCBPolicy as _GaussianBayesianUCBPolicy,
)
from pymab.policies._native_mixin import native_policy_class

BernoulliBayesianUCBPolicy = native_policy_class(
    "bernoulli_bayesian_ucb", _BernoulliBayesianUCBPolicy, module=__name__
)
GaussianBayesianUCBPolicy = native_policy_class(
    "gaussian_bayesian_ucb", _GaussianBayesianUCBPolicy, module=__name__
)

__all__ = ["BernoulliBayesianUCBPolicy", "GaussianBayesianUCBPolicy"]
