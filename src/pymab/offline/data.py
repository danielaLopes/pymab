"""Immutable domain records for offline bandit evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np

from pymab.errors import ValidationError
from pymab.types import FloatArray, IntArray
from pymab.validation import float_array, integer_array, positive_integer

ClusterId = str | int


class TargetPolicy(Protocol):
    """Fixed target decision rule used by off-policy estimators."""

    def probabilities(self, context: FloatArray | None) -> FloatArray:
        """Return one action probability per arm without mutating state."""


@runtime_checkable
class BatchTargetPolicy(Protocol):
    """Optional vectorized target-policy interface for offline estimation."""

    def probabilities_batch(
        self,
        contexts: FloatArray | None,
        *,
        n_events: int,
    ) -> FloatArray:
        """Return a finite ``(event, arm)`` probability matrix."""


class CrossFittedRewardModel(Protocol):
    """Out-of-fold reward predictions used by doubly robust estimation."""

    def predict_event(self, event_index: int, context: FloatArray | None) -> FloatArray:
        """Return expected rewards for every arm at one event."""


class EstimateMethod(StrEnum):
    """Supported fixed-policy off-policy estimators."""

    IPS = "ips"
    SNIPS = "snips"
    DOUBLY_ROBUST = "dr"


class OverlapStatus(StrEnum):
    """Strength of logged-action support for a target policy."""

    ADEQUATE = "adequate"
    WEAK = "weak"
    NONE = "none"
    MODEL_ONLY = "model_only"


class ResamplingUnit(StrEnum):
    """Independent unit used by bootstrap uncertainty estimation."""

    EVENT = "event"
    CLUSTER = "cluster"


class LoggingScheme(StrEnum):
    """Logging design used by sequential replay."""

    UNIFORM = "uniform"
    NONUNIFORM = "nonuniform"


@dataclass(frozen=True, eq=False)
class LoggedBanditDataset:
    """Validated immutable logged contextual-bandit observations."""

    actions: IntArray
    rewards: FloatArray
    propensities: FloatArray
    n_arms: int
    contexts: FloatArray | None = None
    clusters: Sequence[ClusterId] | None = None

    def __post_init__(self) -> None:
        n_arms = positive_integer(self.n_arms, name="n_arms")
        actions = integer_array(
            self.actions,
            name="actions",
            ndim=1,
            minimum=0,
            maximum_exclusive=n_arms,
            readonly=True,
        )
        rewards = float_array(self.rewards, name="rewards", ndim=1, readonly=True)
        propensities = float_array(
            self.propensities,
            name="propensities",
            ndim=1,
            readonly=True,
        )
        if rewards.shape != actions.shape or propensities.shape != actions.shape:
            raise ValidationError(
                "actions, rewards, and propensities must have equal shape"
            )
        if np.any((propensities <= 0) | (propensities > 1)):
            raise ValidationError("propensities must be in (0, 1]")
        contexts: FloatArray | None = None
        if self.contexts is not None:
            contexts = float_array(
                self.contexts,
                name="contexts",
                ndim=np.asarray(self.contexts).ndim,
                readonly=True,
            )
            if contexts.ndim < 2 or contexts.shape[0] != actions.size:
                raise ValidationError("contexts must have one non-scalar row per event")
        clusters = _validate_clusters(self.clusters, n_events=actions.size)
        object.__setattr__(self, "n_arms", n_arms)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "propensities", propensities)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "clusters", clusters)

    @property
    def n_events(self) -> int:
        return int(self.actions.size)

    def context_at(self, index: int) -> FloatArray | None:
        if self.contexts is None:
            return None
        result = np.asarray(self.contexts[index], dtype=float)
        result.flags.writeable = False
        return result

    def equals(self, other: object) -> bool:
        """Return value equality using explicit NumPy semantics."""

        return (
            isinstance(other, LoggedBanditDataset)
            and self.n_arms == other.n_arms
            and self.clusters == other.clusters
            and np.array_equal(self.actions, other.actions)
            and np.array_equal(self.rewards, other.rewards)
            and np.array_equal(self.propensities, other.propensities)
            and (
                self.contexts is None
                and other.contexts is None
                or self.contexts is not None
                and other.contexts is not None
                and np.array_equal(self.contexts, other.contexts)
            )
        )


@dataclass(frozen=True)
class WeightDiagnostics:
    """Raw and post-clipping importance-weight diagnostics."""

    raw_effective_sample_size: float
    effective_sample_size: float
    raw_max_weight: float
    max_weight: float
    raw_mean_weight: float
    mean_weight: float
    clipped_fraction: float
    clipping_threshold: float | None


@dataclass(frozen=True)
class OfflineEstimate:
    """Off-policy estimate with overlap and uncertainty diagnostics."""

    method: EstimateMethod
    estimate: float
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    weights: WeightDiagnostics
    overlap_status: OverlapStatus
    resampling_unit: ResamplingUnit
    confidence_method: str
    confidence_level: float
    n_events: int

    @property
    def effective_sample_size(self) -> float:
        return self.weights.effective_sample_size

    @property
    def max_weight(self) -> float:
        return self.weights.max_weight

    @property
    def mean_weight(self) -> float:
        return self.weights.mean_weight

    @property
    def clipped_fraction(self) -> float:
        return self.weights.clipped_fraction


@dataclass(frozen=True, eq=False)
class SequentialReplayResult:
    """Accepted events and diagnostics from adaptive sequential replay."""

    selected_actions: IntArray
    accepted_event_indices: IntArray
    accepted_actions: IntArray
    accepted_rewards: FloatArray
    logging_scheme: LoggingScheme
    acceptance_scale: float | None

    def __post_init__(self) -> None:
        selected = integer_array(
            self.selected_actions,
            name="selected_actions",
            ndim=1,
            readonly=True,
        )
        indices = integer_array(
            self.accepted_event_indices,
            name="accepted_event_indices",
            ndim=1,
            allow_empty=True,
            minimum=0,
            readonly=True,
        )
        accepted = integer_array(
            self.accepted_actions,
            name="accepted_actions",
            ndim=1,
            allow_empty=True,
            minimum=0,
            readonly=True,
        )
        rewards = float_array(
            self.accepted_rewards,
            name="accepted_rewards",
            ndim=1,
            allow_empty=True,
            readonly=True,
        )
        if not (indices.shape == accepted.shape == rewards.shape):
            raise ValidationError("accepted replay arrays must have equal shapes")
        if accepted.size > selected.size:
            raise ValidationError("accepted events cannot exceed selected events")
        object.__setattr__(self, "selected_actions", selected)
        object.__setattr__(self, "accepted_event_indices", indices)
        object.__setattr__(self, "accepted_actions", accepted)
        object.__setattr__(self, "accepted_rewards", rewards)

    @property
    def n_events(self) -> int:
        return int(self.selected_actions.size)

    @property
    def n_accepted(self) -> int:
        return int(self.accepted_rewards.size)

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_events

    @property
    def average_reward(self) -> float | None:
        if self.n_accepted == 0:
            return None
        return float(np.mean(self.accepted_rewards))

    def equals(self, other: object) -> bool:
        """Return value equality using explicit NumPy semantics."""

        return (
            isinstance(other, SequentialReplayResult)
            and self.logging_scheme == other.logging_scheme
            and self.acceptance_scale == other.acceptance_scale
            and np.array_equal(self.selected_actions, other.selected_actions)
            and np.array_equal(
                self.accepted_event_indices, other.accepted_event_indices
            )
            and np.array_equal(self.accepted_actions, other.accepted_actions)
            and np.array_equal(self.accepted_rewards, other.accepted_rewards)
        )


def _validate_clusters(
    clusters: Sequence[ClusterId] | None, *, n_events: int
) -> tuple[ClusterId, ...] | None:
    if clusters is None:
        return None
    if len(clusters) != n_events:
        raise ValidationError("clusters must contain one identifier per event")
    result: list[ClusterId] = []
    for cluster in clusters:
        if isinstance(cluster, bool) or not isinstance(cluster, (str, Integral)):
            raise ValidationError("cluster identifiers must be strings or integers")
        if isinstance(cluster, str) and not cluster:
            raise ValidationError("cluster identifiers must be non-empty")
        result.append(cluster if isinstance(cluster, str) else int(cluster))
    return tuple(result)


__all__ = [
    "ClusterId",
    "BatchTargetPolicy",
    "CrossFittedRewardModel",
    "EstimateMethod",
    "LoggedBanditDataset",
    "LoggingScheme",
    "OfflineEstimate",
    "OverlapStatus",
    "ResamplingUnit",
    "SequentialReplayResult",
    "TargetPolicy",
    "WeightDiagnostics",
]
