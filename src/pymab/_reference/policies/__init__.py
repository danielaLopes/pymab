"""Pure-Python reference implementations of every built-in policy."""

from pymab._reference.policies.adversarial import EXP3Policy
from pymab._reference.policies.bayesian_ucb import (
    BernoulliBayesianUCBPolicy,
    GaussianBayesianUCBPolicy,
)
from pymab._reference.policies.change_detection import (
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
    PageHinkleyUCBPolicy,
)
from pymab._reference.policies.contextual_bandits import (
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
)
from pymab._reference.policies.epsilon_greedy import (
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
)
from pymab._reference.policies.gradient import GradientBanditPolicy
from pymab._reference.policies.greedy import GreedyPolicy
from pymab._reference.policies.nonstationary import (
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
)
from pymab._reference.policies.pure_exploration import (
    MedianEliminationPolicy,
    SuccessiveEliminationPolicy,
)
from pymab._reference.policies.random import RandomPolicy
from pymab._reference.policies.softmax_selection import SoftmaxPolicy
from pymab._reference.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
)
from pymab._reference.policies.ucb import KLUCBPolicy, MOSSPolicy, UCBPolicy

__all__ = [
    "BernoulliBayesianUCBPolicy",
    "BernoulliThompsonSamplingPolicy",
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
    "GreedyPolicy",
    "KLUCBPolicy",
    "LinUCBPolicy",
    "LinearEpsilonGreedyPolicy",
    "LinearThompsonSamplingPolicy",
    "LogisticContextualBanditPolicy",
    "MOSSPolicy",
    "MedianEliminationPolicy",
    "PageHinkleyUCBPolicy",
    "RandomPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
    "SoftmaxPolicy",
    "SuccessiveEliminationPolicy",
    "UCBPolicy",
]
