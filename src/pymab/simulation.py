"""Reproducible experiment orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import TYPE_CHECKING, cast

import numpy as np

from pymab import _native
from pymab._backend import (
    BackendCompatibilityReport,
    BackendMode,
    compatibility_report,
    reference_policies,
    run_native_experiment,
)
from pymab._experiment import _ExperimentRunner, _RunRequest
from pymab._experiment_storage import _ExperimentStorage
from pymab._random import stable_seed
from pymab._version import __version__
from pymab.environments import ContextualEnvironment, Environment
from pymab.errors import CompatibilityError, ValidationError
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.provenance import JSONValue, RunProvenance
from pymab.validation import positive_integer

if TYPE_CHECKING:
    from pymab.results import SimulationResult


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for independent replicated bandit experiments."""

    horizon: int
    n_replicates: int
    seed: int
    reward_coupling: str = "common"
    record_contexts: bool = False
    backend: BackendMode = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizon", positive_integer(self.horizon, name="horizon")
        )
        object.__setattr__(
            self,
            "n_replicates",
            positive_integer(self.n_replicates, name="n_replicates"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        if self.reward_coupling not in {"common", "independent"}:
            raise ValidationError("reward_coupling must be 'common' or 'independent'")
        if not isinstance(self.record_contexts, bool):
            raise TypeError("record_contexts must be a boolean")
        if self.backend not in {"auto", "rust", "python"}:
            raise ValidationError("backend must be 'auto', 'rust', or 'python'")


class Experiment:
    """Run named policies against a reproducible simulated environment."""

    def __init__(
        self,
        *,
        environment: Environment,
        policies: Mapping[str, Policy | ContextualPolicy],
        config: ExperimentConfig,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        self.environment = environment
        self.policies = _validate_policies(policies)
        self.config = config
        self.metadata = {} if metadata is None else dict(metadata)
        self._validate_compatibility()

    def run(self) -> SimulationResult:
        """Execute every independent replicate and return an immutable result."""

        report = self.backend_compatibility()
        if self.config.backend == "rust" and not report.compatible:
            raise CompatibilityError(report.message())
        use_rust = self.config.backend == "rust" or (
            self.config.backend == "auto" and report.compatible
        )
        if use_rust:
            native_output = run_native_experiment(
                environment=self.environment,
                policies=self.policies,
                horizon=self.config.horizon,
                n_replicates=self.config.n_replicates,
                seed=self.config.seed,
                reward_coupling=self.config.reward_coupling,
                record_contexts=self.config.record_contexts,
            )
            storage = _ExperimentStorage(
                rewards=np.asarray(native_output.rewards, dtype=float),
                actions=np.asarray(native_output.actions, dtype=np.int64),
                expected_rewards=np.asarray(
                    native_output.expected_rewards, dtype=float
                ),
                arm_means=np.asarray(native_output.arm_means, dtype=float),
                optimal_mask=np.asarray(native_output.optimal_mask, dtype=bool),
                recommendations=np.asarray(
                    native_output.recommendations, dtype=np.int64
                ),
                contexts=(
                    None
                    if native_output.contexts is None
                    else np.asarray(native_output.contexts, dtype=float)
                ),
            )
            context_digest = cast(str | None, native_output.context_digest)
            actual_backend = "rust"
            rng_scheme = _native.rng_scheme_version() or "unknown"
        else:
            output = _ExperimentRunner(
                environment=self.environment,
                policies=reference_policies(self.policies),
                request=_RunRequest(
                    horizon=self.config.horizon,
                    n_replicates=self.config.n_replicates,
                    seed=self.config.seed,
                    reward_coupling=self.config.reward_coupling,
                    record_contexts=self.config.record_contexts,
                ),
            ).run()
            storage = output.storage
            context_digest = output.context_digest
            actual_backend = "python"
            rng_scheme = "pymab-v2-blake2b-seedsequence-v1"
        policy_ids = tuple(self.policies)
        provenance = RunProvenance.capture(
            pymab_version=__version__,
            environment=self.environment,
            policies=self.policies,
            backend=actual_backend,
            rng_scheme=rng_scheme,
        )
        from pymab.results import SimulationResult

        return SimulationResult(
            rewards=storage.rewards,
            actions=storage.actions,
            expected_rewards=storage.expected_rewards,
            arm_means=storage.arm_means,
            optimal_mask=storage.optimal_mask,
            recommendations=storage.recommendations,
            contexts=storage.contexts,
            context_digest=context_digest,
            policy_ids=policy_ids,
            replicate_seeds=tuple(
                stable_seed(self.config.seed, replicate, "replicate")
                for replicate in range(self.config.n_replicates)
            ),
            config=cast(Mapping[str, JSONValue], asdict(self.config)),
            metadata=self.metadata,
            provenance=provenance,
            library_version=__version__,
        )

    def backend_compatibility(self) -> BackendCompatibilityReport:
        """Return every reason this experiment cannot use native execution."""

        return compatibility_report(
            environment=self.environment,
            policies=self.policies,
            seed=self.config.seed,
        )

    def _validate_compatibility(self) -> None:
        environment = self.environment
        if not isinstance(environment.contextual, bool):
            raise TypeError("environment.contextual must be a boolean")
        for policy_id, policy in self.policies.items():
            if policy.n_arms != environment.n_arms:
                raise CompatibilityError(
                    f"policy {policy_id!r} n_arms does not match the environment"
                )
            if policy.capabilities.contextual != environment.contextual:
                raise CompatibilityError(
                    f"policy {policy_id!r} contextual mode does not match environment"
                )
            if environment.reward_domain not in policy.capabilities.reward_domains:
                raise CompatibilityError(
                    f"policy {policy_id!r} does not support "
                    f"{environment.reward_domain.value} rewards"
                )
            if isinstance(policy, ContextualPolicy):
                contextual = cast(ContextualEnvironment, environment)
                if policy.n_features != contextual.n_features:
                    raise CompatibilityError(
                        f"policy {policy_id!r} n_features does not match environment"
                    )


def _validate_policies(
    policies: Mapping[str, Policy | ContextualPolicy],
) -> dict[str, Policy | ContextualPolicy]:
    if not policies:
        raise ValidationError("at least one policy is required")
    result: dict[str, Policy | ContextualPolicy] = {}
    for name, policy in policies.items():
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("policy IDs must be non-empty strings")
        if not isinstance(policy, (Policy, ContextualPolicy)):
            raise TypeError(f"policy {name!r} must implement a PyMAB policy contract")
        result[name] = policy
    return result


__all__ = [
    "BackendCompatibilityReport",
    "BackendMode",
    "Experiment",
    "ExperimentConfig",
]
