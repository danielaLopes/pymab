"""Validated immutable simulation results and derived metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from pymab.errors import ValidationError
from pymab.provenance import JSONValue, RunProvenance, freeze_json, thaw_json
from pymab.types import BoolArray, FloatArray, IntArray
from pymab.validation import boolean_array, float_array, integer_array

SCHEMA_VERSION = 3
TIE_RTOL = 1e-12
TIE_ATOL = 1e-12


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Read-only observations, ground truth, configuration, and provenance."""

    rewards: FloatArray
    actions: IntArray
    expected_rewards: FloatArray
    arm_means: FloatArray
    optimal_mask: BoolArray
    recommendations: IntArray
    policy_ids: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    config: Mapping[str, JSONValue]
    provenance: RunProvenance
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    contexts: FloatArray | None = None
    context_digest: str | None = None
    schema_version: int = SCHEMA_VERSION
    library_version: str = ""

    def __post_init__(self) -> None:
        arrays = _ResultArrays.from_values(
            rewards=self.rewards,
            actions=self.actions,
            expected_rewards=self.expected_rewards,
            arm_means=self.arm_means,
            optimal_mask=self.optimal_mask,
            recommendations=self.recommendations,
            contexts=self.contexts,
        )
        arrays.validate_relationships(
            policy_ids=self.policy_ids,
            replicate_seeds=self.replicate_seeds,
        )
        policy_ids = _validate_policy_ids(self.policy_ids, arrays.n_policies)
        replicate_seeds = _validate_replicate_seeds(
            self.replicate_seeds, arrays.n_replicates
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported result schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not isinstance(self.provenance, RunProvenance):
            raise TypeError("provenance must be a RunProvenance")
        if self.context_digest is not None and (
            not isinstance(self.context_digest, str) or not self.context_digest
        ):
            raise ValidationError("context_digest must be a non-empty string")

        object.__setattr__(self, "rewards", arrays.rewards)
        object.__setattr__(self, "actions", arrays.actions)
        object.__setattr__(self, "expected_rewards", arrays.expected_rewards)
        object.__setattr__(self, "arm_means", arrays.arm_means)
        object.__setattr__(self, "optimal_mask", arrays.optimal_mask)
        object.__setattr__(self, "recommendations", arrays.recommendations)
        object.__setattr__(self, "contexts", arrays.contexts)
        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(self, "replicate_seeds", replicate_seeds)
        object.__setattr__(
            self,
            "config",
            cast(Mapping[str, JSONValue], freeze_json(self.config, name="config")),
        )
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, JSONValue], freeze_json(self.metadata, name="metadata")),
        )
        if not self.library_version:
            object.__setattr__(self, "library_version", self.provenance.pymab_version)

    @property
    def n_replicates(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def n_policies(self) -> int:
        return int(self.rewards.shape[2])

    @property
    def n_arms(self) -> int:
        return int(self.arm_means.shape[2])

    @property
    def optimal_values(self) -> FloatArray:
        return np.asarray(np.max(self.arm_means, axis=2), dtype=float)

    @property
    def regret(self) -> FloatArray:
        return np.asarray(
            self.optimal_values[:, :, np.newaxis] - self.expected_rewards,
            dtype=float,
        )

    @property
    def simple_regret(self) -> FloatArray:
        recommended_means = np.take_along_axis(
            self.arm_means, self.recommendations, axis=2
        )
        return np.asarray(
            self.optimal_values[:, :, np.newaxis] - recommended_means,
            dtype=float,
        )

    @property
    def cumulative_regret_by_replicate(self) -> FloatArray:
        return np.asarray(np.cumsum(self.regret, axis=1), dtype=float)

    @property
    def cumulative_regret(self) -> FloatArray:
        return np.asarray(
            np.mean(self.cumulative_regret_by_replicate, axis=0), dtype=float
        )

    @property
    def average_reward_by_step(self) -> FloatArray:
        return np.asarray(np.mean(self.rewards, axis=0), dtype=float)

    @property
    def cumulative_reward_by_step(self) -> FloatArray:
        return np.asarray(np.cumsum(self.average_reward_by_step, axis=0), dtype=float)

    @property
    def optimal_action_indicator(self) -> BoolArray:
        return np.asarray(
            np.take_along_axis(self.optimal_mask, self.actions, axis=2), dtype=bool
        )

    @property
    def optimal_action_rate_by_step(self) -> FloatArray:
        return np.asarray(np.mean(self.optimal_action_indicator, axis=0), dtype=float)

    @property
    def recommendation_is_optimal(self) -> BoolArray:
        return np.asarray(
            np.take_along_axis(self.optimal_mask, self.recommendations, axis=2),
            dtype=bool,
        )

    def equals(self, other: object) -> bool:
        """Return value equality using explicit NumPy array semantics."""

        if not isinstance(other, SimulationResult):
            return False
        scalar_equal = (
            self.policy_ids == other.policy_ids
            and self.replicate_seeds == other.replicate_seeds
            and self.schema_version == other.schema_version
            and self.library_version == other.library_version
            and self.context_digest == other.context_digest
            and thaw_json(self.config) == thaw_json(other.config)
            and thaw_json(self.metadata) == thaw_json(other.metadata)
            and self.provenance.equals(other.provenance)
        )
        if not scalar_equal:
            return False
        arrays_equal = all(
            np.array_equal(left, right)
            for left, right in (
                (self.rewards, other.rewards),
                (self.actions, other.actions),
                (self.expected_rewards, other.expected_rewards),
                (self.arm_means, other.arm_means),
                (self.optimal_mask, other.optimal_mask),
                (self.recommendations, other.recommendations),
            )
        )
        contexts_equal = (
            self.contexts is None
            and other.contexts is None
            or self.contexts is not None
            and other.contexts is not None
            and np.array_equal(self.contexts, other.contexts)
        )
        return arrays_equal and contexts_equal

    def to_dict(self) -> dict[str, object]:
        """Return a schema-versioned JSON-compatible payload."""

        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "policy_ids": list(self.policy_ids),
            "replicate_seeds": list(self.replicate_seeds),
            "config": thaw_json(self.config),
            "metadata": thaw_json(self.metadata),
            "provenance": self.provenance.to_dict(),
            "context_digest": self.context_digest,
            "contexts": None if self.contexts is None else self.contexts.tolist(),
            "rewards": self.rewards.tolist(),
            "actions": self.actions.tolist(),
            "expected_rewards": self.expected_rewards.tolist(),
            "arm_means": self.arm_means.tolist(),
            "optimal_mask": self.optimal_mask.tolist(),
            "recommendations": self.recommendations.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SimulationResult:
        """Construct a result through the versioned persistence schema."""

        from pymab.persistence import ResultSerializer

        return ResultSerializer.from_payload(payload)

    def save_npz(self, path: str | Path) -> Path:
        """Atomically save a compressed result archive and return its path."""

        from pymab.persistence import ResultSerializer

        return ResultSerializer.save_npz(self, path)

    @classmethod
    def load_npz(cls, path: str | Path) -> SimulationResult:
        """Load and validate a compressed result archive."""

        from pymab.persistence import ResultSerializer

        return ResultSerializer.load_npz(path)

    def save_json(self, path: str | Path) -> Path:
        """Atomically save a JSON result and return its normalized path."""

        from pymab.persistence import ResultSerializer

        return ResultSerializer.save_json(self, path)

    @classmethod
    def load_json(cls, path: str | Path) -> SimulationResult:
        """Load and validate a JSON result."""

        from pymab.persistence import ResultSerializer

        return ResultSerializer.load_json(path)

    def to_pandas(self) -> Any:
        """Return a vectorized tidy DataFrame; requires ``pymab[analysis]``."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pymab[analysis] to use to_pandas().") from exc

        n_rows = self.n_replicates * self.horizon * self.n_policies
        replicate = np.repeat(
            np.arange(self.n_replicates), self.horizon * self.n_policies
        )
        step = np.tile(
            np.repeat(np.arange(self.horizon), self.n_policies), self.n_replicates
        )
        policy_index = np.tile(
            np.arange(self.n_policies), self.n_replicates * self.horizon
        )
        flat_actions = self.actions.reshape(n_rows)
        return pd.DataFrame(
            {
                "replicate": replicate,
                "replicate_seed": np.asarray(self.replicate_seeds, dtype=object)[
                    replicate
                ],
                "step": step,
                "policy_index": policy_index,
                "policy_id": np.asarray(self.policy_ids)[policy_index],
                "action": flat_actions,
                "reward": self.rewards.reshape(n_rows),
                "expected_reward": self.expected_rewards.reshape(n_rows),
                "regret": self.regret.reshape(n_rows),
                "recommendation": self.recommendations.reshape(n_rows),
                "simple_regret": self.simple_regret.reshape(n_rows),
                "selected_optimal_action": self.optimal_action_indicator.reshape(
                    n_rows
                ),
                "recommended_optimal_action": self.recommendation_is_optimal.reshape(
                    n_rows
                ),
            }
        )


@dataclass(frozen=True, eq=False)
class _ResultArrays:
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

    @property
    def n_replicates(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_policies(self) -> int:
        return int(self.rewards.shape[2])

    def validate_relationships(
        self,
        *,
        policy_ids: Sequence[object],
        replicate_seeds: Sequence[object],
    ) -> None:
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


def _validate_policy_ids(
    policy_ids: Sequence[object], n_policies: int
) -> tuple[str, ...]:
    if len(policy_ids) != n_policies or any(
        not isinstance(policy_id, str) or not policy_id.strip()
        for policy_id in policy_ids
    ):
        raise ValidationError("policy_ids must be non-empty strings")
    result = tuple(cast(str, policy_id) for policy_id in policy_ids)
    if len(set(result)) != len(result):
        raise ValidationError("policy_ids must be unique")
    return result


def _validate_replicate_seeds(
    replicate_seeds: Sequence[object], n_replicates: int
) -> tuple[int, ...]:
    if len(replicate_seeds) != n_replicates:
        raise ValidationError("replicate_seeds must match the replicate dimension")
    result: list[int] = []
    for seed in replicate_seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValidationError("replicate_seeds must contain only integers")
        if int(seed) < 0:
            raise ValidationError("replicate_seeds must be non-negative")
        result.append(int(seed))
    return tuple(result)


__all__ = ["SCHEMA_VERSION", "SimulationResult"]
