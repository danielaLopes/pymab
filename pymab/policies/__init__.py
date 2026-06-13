"""Policy implementations."""

from pymab.policies.adversarial import EXP3Policy
from pymab.policies.bayesian_ucb import (
    BayesianUCBPolicy,
    BernoulliBayesianUCBPolicy,
    GaussianBayesianUCBPolicy,
)
from pymab.policies.contextual_bandits import (
    ContextualBanditPolicy,
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
)
from pymab.policies.epsilon_greedy import (
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
)
from pymab.policies.gradient import GradientBanditPolicy, GradientPolicy
from pymab.policies.greedy import GreedyPolicy
from pymab.policies.policy import ActionValuePolicy, ContextualPolicy, Policy
from pymab.policies.pure_exploration import (
    MedianEliminationPolicy,
    SuccessiveEliminationPolicy,
)
from pymab.policies.random import RandomPolicy
from pymab.policies.softmax_selection import SoftmaxPolicy, SoftmaxSelectionPolicy
from pymab.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    DiscountedBernoulliThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    ThompsonSamplingPolicy,
)
from pymab.policies.ucb import (
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
    DiscountedUCBPolicy,
    KLUCBPolicy,
    MOSSPolicy,
    MOSSUCBPolicy,
    PageHinkleyUCBPolicy,
    SlidingWindowUCBPolicy,
    StationaryUCBPolicy,
    UCBPolicy,
)

__all__ = [
    "ActionValuePolicy",
    "BayesianUCBPolicy",
    "BernoulliBayesianUCBPolicy",
    "BernoulliThompsonSamplingPolicy",
    "ContextualBanditPolicy",
    "ContextualPolicy",
    "CUSUMUCBPolicy",
    "ChangePointUCBPolicy",
    "DecayingEpsilonGreedyPolicy",
    "DiscountedBernoulliThompsonSamplingPolicy",
    "DiscountedUCBPolicy",
    "EpsilonGreedyPolicy",
    "EXP3Policy",
    "GaussianBayesianUCBPolicy",
    "GaussianThompsonSamplingPolicy",
    "GradientBanditPolicy",
    "GradientPolicy",
    "KLUCBPolicy",
    "GreedyPolicy",
    "LinearEpsilonGreedyPolicy",
    "LinearThompsonSamplingPolicy",
    "LinUCBPolicy",
    "LogisticContextualBanditPolicy",
    "MedianEliminationPolicy",
    "MOSSPolicy",
    "MOSSUCBPolicy",
    "PageHinkleyUCBPolicy",
    "Policy",
    "RandomPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
    "SoftmaxPolicy",
    "SoftmaxSelectionPolicy",
    "StationaryUCBPolicy",
    "SuccessiveEliminationPolicy",
    "ThompsonSamplingPolicy",
    "UCBPolicy",
]
