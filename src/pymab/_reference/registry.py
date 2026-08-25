"""Construction registry for the private pure-Python policy backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pymab._reference import policies
from pymab.policies.policy import ContextualPolicy, Policy

ReferencePolicy = Policy | ContextualPolicy
ReferenceFactory = Callable[..., ReferencePolicy]


@dataclass(frozen=True, slots=True)
class ReferencePolicySpec:
    """Factory and immutable constructor-field contract for one built-in policy."""

    factory: ReferenceFactory
    config_fields: tuple[str, ...]


REFERENCE_POLICY_SPECS: Mapping[str, ReferencePolicySpec] = MappingProxyType(
    {
        "bernoulli_bayesian_ucb": ReferencePolicySpec(
            policies.BernoulliBayesianUCBPolicy,
            ("n_arms", "alpha_prior", "beta_prior", "quantile"),
        ),
        "bernoulli_thompson_sampling": ReferencePolicySpec(
            policies.BernoulliThompsonSamplingPolicy,
            ("n_arms", "alpha_prior", "beta_prior"),
        ),
        "change_point_ucb": ReferencePolicySpec(
            policies.ChangePointUCBPolicy,
            (
                "n_arms",
                "initial_value",
                "c",
                "reward_scale",
                "detector",
                "threshold",
                "drift",
                "min_observations",
            ),
        ),
        "cusum_ucb": ReferencePolicySpec(
            policies.CUSUMUCBPolicy,
            (
                "n_arms",
                "initial_value",
                "c",
                "reward_scale",
                "threshold",
                "drift",
                "min_observations",
            ),
        ),
        "decaying_epsilon_greedy": ReferencePolicySpec(
            policies.DecayingEpsilonGreedyPolicy,
            (
                "n_arms",
                "initial_value",
                "initial_epsilon",
                "min_epsilon",
                "decay_rate",
            ),
        ),
        "discounted_bernoulli_thompson_sampling": ReferencePolicySpec(
            policies.DiscountedBernoulliThompsonSamplingPolicy,
            ("n_arms", "alpha_prior", "beta_prior", "discount_factor"),
        ),
        "discounted_ucb": ReferencePolicySpec(
            policies.DiscountedUCBPolicy,
            (
                "n_arms",
                "initial_value",
                "c",
                "reward_scale",
                "discount_factor",
            ),
        ),
        "epsilon_greedy": ReferencePolicySpec(
            policies.EpsilonGreedyPolicy,
            ("n_arms", "initial_value", "epsilon"),
        ),
        "exp3": ReferencePolicySpec(
            policies.EXP3Policy,
            ("n_arms", "gamma", "learning_rate"),
        ),
        "gaussian_bayesian_ucb": ReferencePolicySpec(
            policies.GaussianBayesianUCBPolicy,
            (
                "n_arms",
                "prior_mean",
                "prior_precision",
                "reward_precision",
                "quantile",
            ),
        ),
        "gaussian_thompson_sampling": ReferencePolicySpec(
            policies.GaussianThompsonSamplingPolicy,
            ("n_arms", "prior_mean", "prior_precision", "reward_precision"),
        ),
        "gradient_bandit": ReferencePolicySpec(
            policies.GradientBanditPolicy,
            ("n_arms", "learning_rate", "use_baseline"),
        ),
        "greedy": ReferencePolicySpec(
            policies.GreedyPolicy,
            ("n_arms", "initial_value"),
        ),
        "kl_ucb": ReferencePolicySpec(
            policies.KLUCBPolicy,
            ("n_arms", "initial_value", "c", "tolerance", "max_iterations"),
        ),
        "lin_ucb": ReferencePolicySpec(
            policies.LinUCBPolicy,
            ("n_arms", "n_features", "alpha", "l2"),
        ),
        "linear_epsilon_greedy": ReferencePolicySpec(
            policies.LinearEpsilonGreedyPolicy,
            ("n_arms", "n_features", "epsilon", "learning_rate"),
        ),
        "linear_thompson_sampling": ReferencePolicySpec(
            policies.LinearThompsonSamplingPolicy,
            ("n_arms", "n_features", "exploration_scale", "l2"),
        ),
        "logistic_contextual_bandit": ReferencePolicySpec(
            policies.LogisticContextualBanditPolicy,
            ("n_arms", "n_features", "epsilon", "learning_rate", "l2"),
        ),
        "median_elimination": ReferencePolicySpec(
            policies.MedianEliminationPolicy,
            ("n_arms", "epsilon", "delta"),
        ),
        "moss": ReferencePolicySpec(
            policies.MOSSPolicy,
            ("n_arms", "initial_value", "horizon", "c", "reward_scale"),
        ),
        "page_hinkley_ucb": ReferencePolicySpec(
            policies.PageHinkleyUCBPolicy,
            (
                "n_arms",
                "initial_value",
                "c",
                "reward_scale",
                "threshold",
                "drift",
                "min_observations",
            ),
        ),
        "random": ReferencePolicySpec(policies.RandomPolicy, ("n_arms",)),
        "sliding_window_bernoulli_thompson_sampling": ReferencePolicySpec(
            policies.SlidingWindowBernoulliThompsonSamplingPolicy,
            ("n_arms", "alpha_prior", "beta_prior", "window_size"),
        ),
        "sliding_window_ucb": ReferencePolicySpec(
            policies.SlidingWindowUCBPolicy,
            ("n_arms", "initial_value", "c", "reward_scale", "window_size"),
        ),
        "softmax": ReferencePolicySpec(
            policies.SoftmaxPolicy,
            ("n_arms", "initial_value", "temperature"),
        ),
        "successive_elimination": ReferencePolicySpec(
            policies.SuccessiveEliminationPolicy,
            ("n_arms", "delta", "confidence_scale"),
        ),
        "ucb": ReferencePolicySpec(
            policies.UCBPolicy,
            ("n_arms", "initial_value", "c", "reward_scale"),
        ),
    }
)

def create_reference_policy(
    kind: str, config: Mapping[str, object]
) -> ReferencePolicy:
    """Construct a fresh reference policy from a registered kind and config."""

    try:
        spec = REFERENCE_POLICY_SPECS[kind]
    except KeyError as error:
        raise ValueError(f"unknown reference policy kind: {kind}") from error
    expected = frozenset(spec.config_fields)
    actual = frozenset(config)
    if actual != expected:
        raise ValueError(
            f"configuration fields differ for {kind}: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return spec.factory(**dict(config))


def reference_policy_kind(policy: ReferencePolicy) -> str:
    """Return the stable registered kind for an exact built-in reference type."""

    for kind, spec in REFERENCE_POLICY_SPECS.items():
        if type(policy) is spec.factory:
            return kind
    raise TypeError("custom policies are not registered reference built-ins")


def reference_policy_config(policy: ReferencePolicy) -> Mapping[str, object]:
    """Return a read-only copy of a built-in policy's constructor configuration."""

    kind = reference_policy_kind(policy)
    spec = REFERENCE_POLICY_SPECS[kind]
    return MappingProxyType(
        {field: getattr(policy, field) for field in spec.config_fields}
    )


def clone_reference_policy(policy: ReferencePolicy) -> ReferencePolicy:
    """Construct a fresh reset policy with the same immutable configuration."""

    kind = reference_policy_kind(policy)
    return create_reference_policy(kind, reference_policy_config(policy))


__all__ = [
    "REFERENCE_POLICY_SPECS",
    "ReferencePolicy",
    "ReferencePolicySpec",
    "clone_reference_policy",
    "create_reference_policy",
    "reference_policy_config",
    "reference_policy_kind",
]
