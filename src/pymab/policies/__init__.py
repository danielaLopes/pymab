"""Policy implementations."""

from pymab.policies.adversarial import EXP3Policy
from pymab.policies.bayesian_ucb import (
    BernoulliBayesianUCBPolicy,
    GaussianBayesianUCBPolicy,
)
from pymab.policies.change_detection import (
    ChangePointUCBPolicy,
    CUSUMUCBPolicy,
    PageHinkleyUCBPolicy,
)
from pymab.policies.contextual_bandits import (
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
)
from pymab.policies.epsilon_greedy import (
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
)
from pymab.policies.gradient import GradientBanditPolicy
from pymab.policies.greedy import GreedyPolicy
from pymab.policies.nonstationary import (
    DiscountedBernoulliThompsonSamplingPolicy,
    DiscountedUCBPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy,
)
from pymab.policies.policy import ActionValuePolicy, ContextualPolicy, Policy
from pymab.policies.pure_exploration import (
    MedianEliminationPolicy,
    SuccessiveEliminationPolicy,
)
from pymab.policies.random import RandomPolicy
from pymab.policies.softmax_selection import SoftmaxPolicy
from pymab.policies.thompson_sampling import (
    BernoulliThompsonSamplingPolicy,
    GaussianThompsonSamplingPolicy,
)
from pymab.policies.ucb import (
    KLUCBPolicy,
    MOSSPolicy,
    UCBPolicy,
)

# Dynamic native wrappers directly inherit their private reference counterpart.
# Register the public descendants to preserve the original public hierarchy for
# callers that use ``isinstance`` or ``issubclass``.
GreedyPolicy.register(EpsilonGreedyPolicy)
GreedyPolicy.register(DecayingEpsilonGreedyPolicy)
UCBPolicy.register(KLUCBPolicy)
UCBPolicy.register(MOSSPolicy)
UCBPolicy.register(SlidingWindowUCBPolicy)
UCBPolicy.register(DiscountedUCBPolicy)
UCBPolicy.register(ChangePointUCBPolicy)
UCBPolicy.register(CUSUMUCBPolicy)
UCBPolicy.register(PageHinkleyUCBPolicy)
BernoulliThompsonSamplingPolicy.register(SlidingWindowBernoulliThompsonSamplingPolicy)
BernoulliThompsonSamplingPolicy.register(DiscountedBernoulliThompsonSamplingPolicy)

__all__ = [
    "ActionValuePolicy",
    "BernoulliBayesianUCBPolicy",
    "BernoulliThompsonSamplingPolicy",
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
    "KLUCBPolicy",
    "GreedyPolicy",
    "LinearEpsilonGreedyPolicy",
    "LinearThompsonSamplingPolicy",
    "LinUCBPolicy",
    "LogisticContextualBanditPolicy",
    "MedianEliminationPolicy",
    "MOSSPolicy",
    "PageHinkleyUCBPolicy",
    "Policy",
    "RandomPolicy",
    "SlidingWindowBernoulliThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
    "SoftmaxPolicy",
    "SuccessiveEliminationPolicy",
    "UCBPolicy",
]
