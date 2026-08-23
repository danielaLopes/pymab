"""Sequential replay for adaptive policies and logged bandit feedback."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import cast

import numpy as np

from pymab._random import generator
from pymab.errors import CompatibilityError, ValidationError
from pymab.offline.data import LoggingScheme, SequentialReplayResult
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.types import FloatArray
from pymab.validation import finite_float, float_array, integer_array


def sequential_replay(
    policy: Policy | ContextualPolicy,
    *,
    logged_actions: object,
    logged_rewards: object,
    logging_scheme: LoggingScheme | str,
    propensities: object | None = None,
    contexts: object | None = None,
    acceptance_scale: float | None = None,
    seed: int = 0,
    clone_policy: bool = True,
    reset_policy: bool = True,
) -> SequentialReplayResult:
    """Evaluate an adaptive policy against logged feedback.

    Uniform logs use classical replay: an event is accepted when the policy's
    selected action matches the logged action. Non-uniform logs additionally
    use rejection sampling with probability ``acceptance_scale / propensity``.
    The scale must be no greater than the smallest logging propensity.
    """

    scheme = _logging_scheme(logging_scheme)
    _validate_boolean(clone_policy, name="clone_policy")
    _validate_boolean(reset_policy, name="reset_policy")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    actions = integer_array(
        logged_actions,
        name="logged_actions",
        ndim=1,
        minimum=0,
        maximum_exclusive=policy.n_arms,
    )
    rewards = float_array(logged_rewards, name="logged_rewards", ndim=1)
    if rewards.shape != actions.shape:
        raise ValidationError("logged_actions and logged_rewards must have equal shape")

    contextual = isinstance(policy, ContextualPolicy)
    event_contexts = _validate_contexts(
        contexts,
        n_events=actions.size,
        policy=policy,
        contextual=contextual,
    )
    event_propensities, scale = _validate_logging_design(
        scheme=scheme,
        propensities=propensities,
        n_events=actions.size,
        n_arms=policy.n_arms,
        acceptance_scale=acceptance_scale,
    )
    replay_policy = _prepare_policy(
        policy,
        clone_policy=clone_policy,
        reset_policy=reset_policy,
    )
    action_rng = generator(int(seed), "sequential-replay", "actions")
    acceptance_rng = generator(int(seed), "sequential-replay", "acceptance")

    selected = np.empty(actions.size, dtype=np.int64)
    accepted_indices: list[int] = []
    for index, (logged_action, reward) in enumerate(zip(actions, rewards, strict=True)):
        context = None if event_contexts is None else event_contexts[index]
        action = _select_action(
            replay_policy,
            context=context,
            rng=action_rng,
            event_index=index,
        )
        selected[index] = action
        if action != int(logged_action):
            continue
        if scheme is LoggingScheme.NONUNIFORM:
            acceptance_probability = scale / float(event_propensities[index])
            if acceptance_rng.random() >= acceptance_probability:
                continue
        accepted_indices.append(index)
        _update_policy(
            replay_policy,
            action=action,
            reward=float(reward),
            context=context,
        )

    indices = np.asarray(accepted_indices, dtype=np.int64)
    return SequentialReplayResult(
        selected_actions=selected,
        accepted_event_indices=indices,
        accepted_actions=actions[indices],
        accepted_rewards=rewards[indices],
        logging_scheme=scheme,
        acceptance_scale=scale,
    )


def _logging_scheme(value: LoggingScheme | str) -> LoggingScheme:
    try:
        return LoggingScheme(value)
    except ValueError as exc:
        raise ValidationError(
            "logging_scheme must be 'uniform' or 'nonuniform'"
        ) from exc


def _validate_boolean(value: object, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_contexts(
    contexts: object | None,
    *,
    n_events: int,
    policy: Policy | ContextualPolicy,
    contextual: bool,
) -> FloatArray | None:
    if contextual:
        if contexts is None:
            raise CompatibilityError("contextual replay requires contexts")
        contextual_policy = cast(ContextualPolicy, policy)
        result = float_array(contexts, name="contexts", ndim=3)
        expected = (n_events, contextual_policy.n_arms, contextual_policy.n_features)
        if result.shape != expected:
            raise CompatibilityError(f"contexts must have shape {expected}")
        return result
    if contexts is not None:
        raise CompatibilityError(
            "contexts cannot be supplied to a non-contextual policy"
        )
    return None


def _validate_logging_design(
    *,
    scheme: LoggingScheme,
    propensities: object | None,
    n_events: int,
    n_arms: int,
    acceptance_scale: float | None,
) -> tuple[FloatArray, float]:
    uniform_propensity = 1.0 / n_arms
    if scheme is LoggingScheme.UNIFORM:
        if acceptance_scale is not None:
            raise ValidationError(
                "acceptance_scale is only valid for non-uniform logging"
            )
        if propensities is None:
            values = np.full(n_events, uniform_propensity, dtype=float)
        else:
            values = _propensity_array(propensities, n_events=n_events)
            if not np.allclose(values, uniform_propensity, rtol=1e-12, atol=1e-12):
                raise ValidationError(
                    "uniform logging propensities must all equal 1 / n_arms"
                )
        return values, uniform_propensity

    if propensities is None:
        raise ValidationError("non-uniform logging requires propensities")
    values = _propensity_array(propensities, n_events=n_events)
    minimum = float(np.min(values))
    scale = (
        minimum
        if acceptance_scale is None
        else finite_float(acceptance_scale, name="acceptance_scale")
    )
    if not 0 < scale <= minimum:
        raise ValidationError(
            "acceptance_scale must be positive and no greater than the minimum "
            "logging propensity"
        )
    return values, scale


def _propensity_array(value: object, *, n_events: int) -> FloatArray:
    result = float_array(value, name="propensities", ndim=1)
    if result.shape != (n_events,):
        raise ValidationError("propensities must contain one value per event")
    if np.any((result <= 0) | (result > 1)):
        raise ValidationError("propensities must be in (0, 1]")
    return result


def _prepare_policy(
    policy: Policy | ContextualPolicy,
    *,
    clone_policy: bool,
    reset_policy: bool,
) -> Policy | ContextualPolicy:
    if clone_policy:
        result = policy.clone() if reset_policy else deepcopy(policy)
    else:
        result = policy
        if reset_policy:
            result.reset()
    return result


def _select_action(
    policy: Policy | ContextualPolicy,
    *,
    context: FloatArray | None,
    rng: np.random.Generator,
    event_index: int,
) -> int:
    if isinstance(policy, ContextualPolicy):
        if context is None:
            raise RuntimeError("context validation failed")
        raw_action = policy.select_action(context=context, rng=rng)
    else:
        raw_action = policy.select_action(rng=rng)
    if isinstance(raw_action, bool) or not isinstance(raw_action, Integral):
        raise ValidationError(
            f"policy returned a non-integer action at event {event_index}"
        )
    action = int(raw_action)
    if not 0 <= action < policy.n_arms:
        raise ValidationError(
            f"policy returned action {action} outside [0, {policy.n_arms}) "
            f"at event {event_index}"
        )
    return action


def _update_policy(
    policy: Policy | ContextualPolicy,
    *,
    action: int,
    reward: float,
    context: FloatArray | None,
) -> None:
    if isinstance(policy, ContextualPolicy):
        if context is None:
            raise RuntimeError("context validation failed")
        policy.update(action=action, reward=reward, context=context)
    else:
        policy.update(action=action, reward=reward)


__all__ = ["sequential_replay"]
