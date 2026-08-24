"""Internal deterministic experiment execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import cast

import numpy as np

from pymab._experiment_storage import _AnyHash, _ExperimentStorage
from pymab._random import generator
from pymab.environments import (
    ClassicEnvironment,
    ContextualEnvironment,
    Environment,
)
from pymab.errors import CompatibilityError, ValidationError
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.types import FloatArray
from pymab.validation import float_array


@dataclass(frozen=True)
class _RunRequest:
    """Validated scalar inputs required by the experiment runner."""

    horizon: int
    n_replicates: int
    seed: int
    reward_coupling: str
    record_contexts: bool


@dataclass(frozen=True)
class _RunOutput:
    """Recorded execution data returned to the public facade."""

    storage: _ExperimentStorage
    context_digest: str | None


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
        """Create independent, stable random streams for one replicate."""

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


class _ExperimentRunner:
    """Execute validated experiment components into preallocated storage."""

    def __init__(
        self,
        *,
        environment: Environment,
        policies: Mapping[str, Policy | ContextualPolicy],
        request: _RunRequest,
    ) -> None:
        self._environment = environment
        self._policies = policies
        self._request = request

    def run(self) -> _RunOutput:
        """Run every replicate while preserving deterministic stream isolation."""

        storage = _ExperimentStorage.create(
            n_replicates=self._request.n_replicates,
            horizon=self._request.horizon,
            n_arms=self._environment.n_arms,
            n_policies=len(self._policies),
            n_features=(
                cast(ContextualEnvironment, self._environment).n_features
                if self._environment.contextual and self._request.record_contexts
                else None
            ),
        )
        context_hasher = hashlib.blake2b(digest_size=32, person=b"pymab-context-v2")
        policy_ids = tuple(self._policies)

        for replicate in range(self._request.n_replicates):
            environment = self._environment.clone()
            policies = {name: policy.clone() for name, policy in self._policies.items()}
            streams = _ReplicateStreams.create(
                master_seed=self._request.seed,
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

        return _RunOutput(
            storage=storage,
            context_digest=(
                context_hasher.hexdigest() if self._environment.contextual else None
            ),
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
        for step in range(self._request.horizon):
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
                if self._request.reward_coupling == "common"
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

    @staticmethod
    def _environment_state(
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
            contextual_context = cast(FloatArray, context)
            policy.update(action=action, reward=reward, context=contextual_context)
            recommendation_value = policy.recommend_action(context=contextual_context)
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


__all__: list[str] = []
