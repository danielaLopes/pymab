"""Native compatibility classification and experiment dispatch helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from pymab import _native
from pymab._reference.registry import clone_reference_policy, reference_policy_kind
from pymab.distributions import (
    BernoulliReward,
    GaussianReward,
    RewardModel,
    UniformReward,
)
from pymab.environments import (
    AbruptShift,
    BanditEnvironment,
    Environment,
    FixedContextProvider,
    GaussianContextProvider,
    GradualDrift,
    LinearContextualEnvironment,
    LogisticContextualEnvironment,
    ProbabilityDrift,
    RandomArmSwap,
    StationaryDynamics,
)
from pymab.policies._native_mixin import NativePolicyMixin
from pymab.policies.policy import ContextualPolicy, Policy

BackendMode = Literal["auto", "rust", "python"]


@dataclass(frozen=True)
class BackendCompatibilityReport:
    """Complete explanation of whether an experiment can run natively."""

    issues: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Return whether no native compatibility issues were found."""

        return not self.issues

    def message(self) -> str:
        """Return one aggregated diagnostic suitable for an exception."""

        if self.compatible:
            return "all experiment components are native-compatible"
        return "native execution is unavailable:\n- " + "\n- ".join(self.issues)


def compatibility_report(
    *,
    environment: Environment,
    policies: Mapping[str, Policy | ContextualPolicy],
    seed: int,
) -> BackendCompatibilityReport:
    """Classify every component that can prevent native execution."""

    issues: list[str] = []
    if not _native.native_available():
        issues.append("the compiled pymab._pymab extension is not importable")
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        issues.append("seed must be in [0, 2**64 - 1] for native execution")
    try:
        _environment_configuration(environment)
    except TypeError as error:
        issues.append(str(error))
    for policy_id, policy in policies.items():
        if not isinstance(policy, NativePolicyMixin):
            issues.append(
                f"policy {policy_id!r} ({type(policy).__name__}) is not a native built-in"
            )
    return BackendCompatibilityReport(tuple(issues))


def run_native_experiment(
    *,
    environment: Environment,
    policies: Mapping[str, Policy | ContextualPolicy],
    horizon: int,
    n_replicates: int,
    seed: int,
    reward_coupling: str,
    record_contexts: bool,
) -> Any:
    """Execute one already-classified native experiment."""

    environment_handle = _native.create_environment(
        json.dumps(
            _environment_configuration(environment),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    native_policies: list[tuple[str, Any]] = []
    for policy_id, policy in policies.items():
        if not isinstance(policy, NativePolicyMixin):
            raise TypeError(f"policy {policy_id!r} is not native")
        native_policies.append((policy_id, policy._native_handle))
    return _native.run_experiment(
        environment_handle,
        native_policies,
        horizon,
        n_replicates,
        seed,
        reward_coupling,
        record_contexts,
    )


def reference_policies(
    policies: Mapping[str, Policy | ContextualPolicy],
) -> dict[str, Policy | ContextualPolicy]:
    """Return fresh reference policies for built-ins and preserve custom policies."""

    result: dict[str, Policy | ContextualPolicy] = {}
    for policy_id, policy in policies.items():
        try:
            reference_policy_kind(policy)
        except TypeError:
            result[policy_id] = policy
        else:
            result[policy_id] = clone_reference_policy(policy)
    return result


def _reward_configuration(model: RewardModel) -> dict[str, object]:
    if type(model) is GaussianReward:
        return {"kind": "gaussian", "std": model.std}
    if type(model) is BernoulliReward:
        return {"kind": "bernoulli"}
    if type(model) is UniformReward:
        return {
            "kind": "uniform",
            "half_width": model.half_width,
        }
    raise TypeError(
        f"reward model {type(model).__name__} is custom; use a built-in reward model"
    )


def _dynamics_configuration(dynamics: object) -> dict[str, object]:
    if type(dynamics) is StationaryDynamics:
        return {"kind": "stationary"}
    if type(dynamics) is GradualDrift:
        return {"kind": "gradual", "std": dynamics.std}
    if type(dynamics) is AbruptShift:
        abrupt = dynamics
        return {
            "kind": "abrupt",
            "frequency": abrupt.frequency,
            "std": abrupt.std,
            "shift_at_step_zero": abrupt.shift_at_step_zero,
        }
    if type(dynamics) is ProbabilityDrift:
        probability_drift = dynamics
        return {
            "kind": "probability",
            "logit_std": probability_drift.logit_std,
            "epsilon": probability_drift.epsilon,
        }
    if type(dynamics) is RandomArmSwap:
        return {
            "kind": "random_swap",
            "probability": dynamics.probability,
        }
    raise TypeError(
        f"dynamics {type(dynamics).__name__} is custom; use built-in dynamics"
    )


def _context_provider_configuration(
    provider: object, *, n_arms: int, n_features: int
) -> dict[str, object]:
    if type(provider) is FixedContextProvider:
        fixed_value = provider.value
        if fixed_value.shape not in {(n_features,), (n_arms, n_features)}:
            raise TypeError(
                "fixed context provider shape does not match the environment"
            )
        return {"kind": "fixed", "values": fixed_value.reshape(-1).tolist()}
    if type(provider) is GaussianContextProvider:
        gaussian = provider
        if gaussian.n_arms != n_arms or gaussian.n_features != n_features:
            raise TypeError(
                "Gaussian context provider shape does not match the environment"
            )
        return {"kind": "gaussian", "mean": gaussian.mean, "std": gaussian.std}
    raise TypeError(
        f"context provider {type(provider).__name__} requires Python callbacks; "
        "use FixedContextProvider or GaussianContextProvider"
    )


def _environment_configuration(environment: Environment) -> dict[str, object]:
    if type(environment) is BanditEnvironment:
        classic = environment
        return {
            "kind": "classic",
            "means": classic.means.tolist(),
            "reward": _reward_configuration(classic.reward_model),
            "dynamics": _dynamics_configuration(classic.dynamics),
        }
    if type(environment) in {
        LinearContextualEnvironment,
        LogisticContextualEnvironment,
    }:
        contextual = cast(LinearContextualEnvironment, environment)
        return {
            "kind": (
                "logistic"
                if type(environment) is LogisticContextualEnvironment
                else "linear"
            ),
            "n_arms": contextual.n_arms,
            "n_features": contextual.n_features,
            "theta": contextual.theta.reshape(-1).tolist(),
            "context_provider": _context_provider_configuration(
                contextual.context_provider,
                n_arms=contextual.n_arms,
                n_features=contextual.n_features,
            ),
            "reward": _reward_configuration(contextual.reward_model),
        }
    raise TypeError(
        f"environment {type(environment).__name__} is custom; use a built-in environment"
    )


__all__ = ["BackendCompatibilityReport", "BackendMode"]
