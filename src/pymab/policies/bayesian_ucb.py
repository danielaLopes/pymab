"""Compatibility imports for the Python reference Bayesian-UCB policies."""

from pymab._reference.policies.bayesian_ucb import (
    BernoulliBayesianUCBPolicy,
    GaussianBayesianUCBPolicy,
)

__all__ = ["BernoulliBayesianUCBPolicy", "GaussianBayesianUCBPolicy"]
