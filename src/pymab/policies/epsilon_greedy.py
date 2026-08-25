"""Compatibility imports for Python reference epsilon-greedy policies."""

from pymab._reference.policies.epsilon_greedy import (
    DecayingEpsilonGreedyPolicy as _DecayingEpsilonGreedyPolicy,
)
from pymab._reference.policies.epsilon_greedy import (
    EpsilonGreedyPolicy as _EpsilonGreedyPolicy,
)
from pymab.policies._native_mixin import native_policy_class

EpsilonGreedyPolicy = native_policy_class(
    "epsilon_greedy", _EpsilonGreedyPolicy, module=__name__
)
DecayingEpsilonGreedyPolicy = native_policy_class(
    "decaying_epsilon_greedy", _DecayingEpsilonGreedyPolicy, module=__name__
)

__all__ = ["DecayingEpsilonGreedyPolicy", "EpsilonGreedyPolicy"]
