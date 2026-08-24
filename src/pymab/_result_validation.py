"""Array normalization and cross-field invariants for simulation results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from pymab.errors import ValidationError
from pymab.types import BoolArray, FloatArray, IntArray
from pymab.validation import boolean_array, float_array, integer_array

TIE_RTOL = 1e-12
TIE_ATOL = 1e-12


@dataclass(frozen=True, eq=False)
class _ResultArrays:
    """Owned, read-only result arrays with validated relationships."""

    rewards: FloatArray
    actions: IntArray
    expected_rewards: FloatArray
    arm_means: FloatArray
    optimal_mask: BoolArray
    recommendations: IntArray
    contexts: FloatArray | None

    @classmethod
    def from_values(
        cls,
        *,
        rewards: object,
        actions: object,
        expected_rewards: object,
        arm_means: object,
        optimal_mask: object,
        recommendations: object,
        contexts: object | None,
    ) -> _ResultArrays:
        """Normalize result values into owned, immutable arrays."""

        return cls(
            rewards=float_array(rewards, name="rewards", ndim=3, readonly=True),
            actions=integer_array(actions, name="actions", ndim=3, readonly=True),
            expected_rewards=float_array(
                expected_rewards,
                name="expected_rewards",
                ndim=3,
                readonly=True,
            ),
            arm_means=float_array(arm_means, name="arm_means", ndim=3, readonly=True),
            optimal_mask=boolean_array(
                optimal_mask, name="optimal_mask", ndim=3, readonly=True
            ),
            recommendations=integer_array(
                recommendations, name="recommendations", ndim=3, readonly=True
            ),
            contexts=(
                None
                if contexts is None
                else float_array(contexts, name="contexts", ndim=4, readonly=True)
            ),
        )

    def validate_relationships(
        self,
        *,
        policy_ids: Sequence[object],
        replicate_seeds: Sequence[object],
    ) -> None:
        """Validate dimensions, indices, masks, and selected arm means."""

        if (
            self.rewards.shape != self.actions.shape
            or self.rewards.shape != self.expected_rewards.shape
        ):
            raise ValidationError(
                "rewards, actions, and expected_rewards must have equal shapes"
            )
        if self.recommendations.shape != self.actions.shape:
            raise ValidationError("recommendations must match actions shape")
        n_replicates, horizon, n_policies = self.rewards.shape
        if self.arm_means.shape[:2] != (n_replicates, horizon):
            raise ValidationError("arm_means must match replicate and step dimensions")
        if self.optimal_mask.shape != self.arm_means.shape:
            raise ValidationError("optimal_mask must match arm_means shape")
        if (
            self.contexts is not None
            and self.contexts.shape[:3] != self.arm_means.shape
        ):
            raise ValidationError(
                "contexts must have shape (replicate, step, arm, feature)"
            )
        if len(policy_ids) != n_policies:
            raise ValidationError("policy_ids must match the policy dimension")
        if len(replicate_seeds) != n_replicates:
            raise ValidationError("replicate_seeds must match the replicate dimension")
        if not np.all(np.any(self.optimal_mask, axis=2)):
            raise ValidationError("every step must have at least one optimal arm")
        n_arms = self.arm_means.shape[2]
        if np.any((self.actions < 0) | (self.actions >= n_arms)):
            raise ValidationError("actions contain an invalid arm index")
        if np.any((self.recommendations < 0) | (self.recommendations >= n_arms)):
            raise ValidationError("recommendations contain an invalid arm index")
        selected_means = np.take_along_axis(self.arm_means, self.actions, axis=2)
        if not np.allclose(
            selected_means,
            self.expected_rewards,
            rtol=TIE_RTOL,
            atol=TIE_ATOL,
        ):
            raise ValidationError("expected_rewards do not match selected arm means")


def _validate_policy_ids(policy_ids: Sequence[object]) -> tuple[str, ...]:
    if any(
        not isinstance(policy_id, str) or not policy_id.strip()
        for policy_id in policy_ids
    ):
        raise ValidationError("policy_ids must be non-empty strings")
    result = tuple(cast(str, policy_id) for policy_id in policy_ids)
    if len(set(result)) != len(result):
        raise ValidationError("policy_ids must be unique")
    return result


def _validate_replicate_seeds(replicate_seeds: Sequence[object]) -> tuple[int, ...]:
    result: list[int] = []
    for seed in replicate_seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValidationError("replicate_seeds must contain only integers")
        if int(seed) < 0:
            raise ValidationError("replicate_seeds must be non-negative")
        result.append(int(seed))
    return tuple(result)


__all__: list[str] = []
