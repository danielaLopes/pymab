"""Reproducible experiment orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Protocol, cast

import numpy as np

from pymab._random import generator, stable_seed
from pymab.environments import (
    ClassicEnvironment,
    ContextualEnvironment,
    Environment,
)
from pymab.errors import CompatibilityError, ValidationError
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.provenance import JSONValue, RunProvenance
from pymab.results import TIE_ATOL, TIE_RTOL, SimulationResult
from pymab.types import FloatArray
from pymab.validation import float_array, positive_integer


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for independent replicated bandit experiments."""

    horizon: int
    n_replicates: int
    seed: int
    reward_coupling: str = "common"
    record_contexts: bool = False

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

        storage = _ExperimentStorage.create(
            config=self.config,
            n_arms=self.environment.n_arms,
            n_policies=len(self.policies),
            n_features=(
                cast(ContextualEnvironment, self.environment).n_features
                if self.environment.contextual and self.config.record_contexts
                else None
            ),
        )
        context_hasher = hashlib.blake2b(digest_size=32, person=b"pymab-context-v2")
        policy_ids = tuple(self.policies)

        for replicate in range(self.config.n_replicates):
            environment = self.environment.clone()
            policies = {name: policy.clone() for name, policy in self.policies.items()}
            streams = _ReplicateStreams.create(
                master_seed=self.config.seed,
                replicate=replicate,
                policy_ids=policy_ids,
            )
            self._run_replicate(
                replicate=replicate,
                environment=environment,
                policies=policies,
                streams=streams,
                storage=storage,
                context_hasher=context_hasher,
            )

        from pymab import __version__

        provenance = RunProvenance.capture(
            pymab_version=__version__,
            environment=self.environment,
            policies=self.policies,
        )
        return SimulationResult(
            rewards=storage.rewards,
            actions=storage.actions,
            expected_rewards=storage.expected_rewards,
            arm_means=storage.arm_means,
            optimal_mask=storage.optimal_mask,
            recommendations=storage.recommendations,
            contexts=storage.contexts,
            context_digest=(
                context_hasher.hexdigest() if self.environment.contextual else None
            ),
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

    def _run_replicate(
        self,
        *,
        replicate: int,
        environment: Environment,
        policies: Mapping[str, Policy | ContextualPolicy],
        streams: _ReplicateStreams,
        storage: _ExperimentStorage,
        context_hasher: _AnyHash,
    ) -> None:
        for step in range(self.config.horizon):
            context, means = self._environment_state(
                environment=environment,
                step=step,
                streams=streams,
            )
            storage.record_environment(
                replicate=replicate,
                step=step,
                means=means,
                context=context,
                context_hasher=context_hasher,
            )
            common_rewards = (
                self._sample_rewards(
                    environment=environment,
                    means=means,
                    rng=streams.common_reward,
                    replicate=replicate,
                    step=step,
                )
                if self.config.reward_coupling == "common"
                else None
            )
            for policy_index, policy_id in enumerate(policies):
                self._run_policy_step(
                    replicate=replicate,
                    step=step,
                    policy_index=policy_index,
                    policy_id=policy_id,
                    policy=policies[policy_id],
                    environment=environment,
                    context=context,
                    means=means,
                    potential_rewards=common_rewards,
                    streams=streams,
                    storage=storage,
                )

    def _environment_state(
        self,
        *,
        environment: Environment,
        step: int,
        streams: _ReplicateStreams,
    ) -> tuple[FloatArray | None, FloatArray]:
        if environment.contextual:
            contextual = cast(ContextualEnvironment, environment)
            context = contextual.context(streams.context)
            means = contextual.expected_rewards(context)
            return context, _validate_means(means, n_arms=environment.n_arms)
        classic = cast(ClassicEnvironment, environment)
        classic.advance(step=step, rng=streams.dynamics)
        return None, _validate_means(
            classic.expected_rewards(), n_arms=environment.n_arms
        )

    def _run_policy_step(
        self,
        *,
        replicate: int,
        step: int,
        policy_index: int,
        policy_id: str,
        policy: Policy | ContextualPolicy,
        environment: Environment,
        context: FloatArray | None,
        means: FloatArray,
        potential_rewards: FloatArray | None,
        streams: _ReplicateStreams,
        storage: _ExperimentStorage,
    ) -> None:
        if isinstance(policy, ContextualPolicy):
            if context is None:
                raise CompatibilityError(
                    f"policy {policy_id!r} requires context at replicate "
                    f"{replicate}, step {step}"
                )
            action_value = policy.select_action(
                context=context, rng=streams.action[policy_id]
            )
        else:
            action_value = policy.select_action(rng=streams.action[policy_id])
        action = _validate_policy_action(
            action_value,
            n_arms=environment.n_arms,
            policy_id=policy_id,
            replicate=replicate,
            step=step,
            field="selected action",
        )
        rewards = (
            potential_rewards
            if potential_rewards is not None
            else self._sample_rewards(
                environment=environment,
                means=means,
                rng=streams.reward[policy_id],
                replicate=replicate,
                step=step,
            )
        )
        reward = float(rewards[action])
        if isinstance(policy, ContextualPolicy):
            if context is None:
                raise CompatibilityError("context validation failed")
            policy.update(action=action, reward=reward, context=context)
            recommendation_value = policy.recommend_action(context=context)
        else:
            policy.update(action=action, reward=reward)
            recommendation_value = policy.recommend_action()
        recommendation = _validate_policy_action(
            recommendation_value,
            n_arms=environment.n_arms,
            policy_id=policy_id,
            replicate=replicate,
            step=step,
            field="recommendation",
        )
        storage.record_policy(
            replicate=replicate,
            step=step,
            policy_index=policy_index,
            action=action,
            reward=reward,
            expected_reward=float(means[action]),
            recommendation=recommendation,
        )

    @staticmethod
    def _sample_rewards(
        *,
        environment: Environment,
        means: FloatArray,
        rng: np.random.Generator,
        replicate: int,
        step: int,
    ) -> FloatArray:
        try:
            rewards = float_array(
                environment.reward_model.sample(means, rng),
                name="potential rewards",
                ndim=1,
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid reward sample at replicate {replicate}, step {step}: {exc}"
            ) from exc
        if rewards.shape != means.shape:
            raise ValidationError(
                f"reward model returned shape {rewards.shape} at replicate "
                f"{replicate}, step {step}; expected {means.shape}"
            )
        return rewards

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


class _AnyHash(Protocol):
    """Minimal structural type for hashlib objects used during a run."""

    def update(self, data: bytes) -> None: ...


@dataclass(frozen=True)
class _ReplicateStreams:
    dynamics: np.random.Generator
    context: np.random.Generator
    common_reward: np.random.Generator
    action: Mapping[str, np.random.Generator]
    reward: Mapping[str, np.random.Generator]

    @classmethod
    def create(
        cls, *, master_seed: int, replicate: int, policy_ids: tuple[str, ...]
    ) -> _ReplicateStreams:
        return cls(
            dynamics=generator(master_seed, replicate, "dynamics"),
            context=generator(master_seed, replicate, "context"),
            common_reward=generator(master_seed, replicate, "reward", "common"),
            action={
                name: generator(master_seed, replicate, "action", name)
                for name in policy_ids
            },
            reward={
                name: generator(master_seed, replicate, "reward", name)
                for name in policy_ids
            },
        )


@dataclass(eq=False)
class _ExperimentStorage:
    rewards: FloatArray
    actions: np.ndarray
    expected_rewards: FloatArray
    arm_means: FloatArray
    optimal_mask: np.ndarray
    recommendations: np.ndarray
    contexts: FloatArray | None

    @classmethod
    def create(
        cls,
        *,
        config: ExperimentConfig,
        n_arms: int,
        n_policies: int,
        n_features: int | None,
    ) -> _ExperimentStorage:
        shape = (config.n_replicates, config.horizon, n_policies)
        contexts = (
            None
            if n_features is None
            else np.empty(
                (config.n_replicates, config.horizon, n_arms, n_features),
                dtype=float,
            )
        )
        return cls(
            rewards=np.empty(shape, dtype=float),
            actions=np.empty(shape, dtype=np.int64),
            expected_rewards=np.empty(shape, dtype=float),
            arm_means=np.empty(
                (config.n_replicates, config.horizon, n_arms), dtype=float
            ),
            optimal_mask=np.empty(
                (config.n_replicates, config.horizon, n_arms), dtype=bool
            ),
            recommendations=np.empty(shape, dtype=np.int64),
            contexts=contexts,
        )

    def record_environment(
        self,
        *,
        replicate: int,
        step: int,
        means: FloatArray,
        context: FloatArray | None,
        context_hasher: _AnyHash,
    ) -> None:
        self.arm_means[replicate, step] = means
        best = float(np.max(means))
        self.optimal_mask[replicate, step] = np.isclose(
            means, best, rtol=TIE_RTOL, atol=TIE_ATOL
        )
        if context is not None:
            contiguous = np.ascontiguousarray(context, dtype=np.float64)
            context_hasher.update(
                np.asarray(contiguous.shape, dtype=np.int64).tobytes()
            )
            context_hasher.update(contiguous.tobytes())
            if self.contexts is not None:
                self.contexts[replicate, step] = contiguous

    def record_policy(
        self,
        *,
        replicate: int,
        step: int,
        policy_index: int,
        action: int,
        reward: float,
        expected_reward: float,
        recommendation: int,
    ) -> None:
        index = (replicate, step, policy_index)
        self.actions[index] = action
        self.rewards[index] = reward
        self.expected_rewards[index] = expected_reward
        self.recommendations[index] = recommendation


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


def _validate_means(value: object, *, n_arms: int) -> FloatArray:
    means = float_array(value, name="expected rewards", ndim=1)
    if means.shape != (n_arms,):
        raise ValidationError("environment must return one expected reward per arm")
    return means


def _validate_policy_action(
    value: object,
    *,
    n_arms: int,
    policy_id: str,
    replicate: int,
    step: int,
    field: str,
) -> int:
    location = f"policy {policy_id!r} {field} at replicate {replicate}, step {step}"
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{location} must be an integer; received {value!r}")
    action = int(value)
    if not 0 <= action < n_arms:
        raise ValidationError(f"{location} {action} is outside [0, {n_arms})")
    return action


__all__ = ["Experiment", "ExperimentConfig", "SimulationResult"]
