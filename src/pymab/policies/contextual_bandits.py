"""Compatibility imports for Python reference contextual policies."""

from pymab._reference.policies.contextual_bandits import (
    LinearEpsilonGreedyPolicy as _LinearEpsilonGreedyPolicy,
)
from pymab._reference.policies.contextual_bandits import (
    LinearThompsonSamplingPolicy as _LinearThompsonSamplingPolicy,
)
from pymab._reference.policies.contextual_bandits import (
    LinUCBPolicy as _LinUCBPolicy,
)
from pymab._reference.policies.contextual_bandits import (
    LogisticContextualBanditPolicy as _LogisticContextualBanditPolicy,
)
from pymab.policies._native_mixin import native_policy_class

LinearEpsilonGreedyPolicy = native_policy_class(
    "linear_epsilon_greedy", _LinearEpsilonGreedyPolicy, module=__name__
)
LinUCBPolicy = native_policy_class("lin_ucb", _LinUCBPolicy, module=__name__)
LinearThompsonSamplingPolicy = native_policy_class(
    "linear_thompson_sampling", _LinearThompsonSamplingPolicy, module=__name__
)
LogisticContextualBanditPolicy = native_policy_class(
    "logistic_contextual_bandit", _LogisticContextualBanditPolicy, module=__name__
)

__all__ = [
    "LinearEpsilonGreedyPolicy",
    "LinearThompsonSamplingPolicy",
    "LinUCBPolicy",
    "LogisticContextualBanditPolicy",
]
