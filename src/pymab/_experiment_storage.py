"""Preallocated storage for experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pymab._result_validation import TIE_ATOL, TIE_RTOL
from pymab.types import FloatArray


class _AnyHash(Protocol):
    """Minimal structural type for hash objects used during a run."""

    def update(self, data: bytes) -> None: ...


@dataclass(eq=False)
class _ExperimentStorage:
    """Mutable arrays populated by the internal experiment runner."""

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
        n_replicates: int,
        horizon: int,
        n_arms: int,
        n_policies: int,
        n_features: int | None,
    ) -> _ExperimentStorage:
        """Allocate all result arrays for an experiment run."""

        shape = (n_replicates, horizon, n_policies)
        contexts = (
            None
            if n_features is None
            else np.empty(
                (n_replicates, horizon, n_arms, n_features),
                dtype=float,
            )
        )
        return cls(
            rewards=np.empty(shape, dtype=float),
            actions=np.empty(shape, dtype=np.int64),
            expected_rewards=np.empty(shape, dtype=float),
            arm_means=np.empty((n_replicates, horizon, n_arms), dtype=float),
            optimal_mask=np.empty((n_replicates, horizon, n_arms), dtype=bool),
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
        """Record ground truth and optional contextual state for one step."""

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
        """Record one policy observation for a replicate step."""

        index = (replicate, step, policy_index)
        self.actions[index] = action
        self.rewards[index] = reward
        self.expected_rewards[index] = expected_reward
        self.recommendations[index] = recommendation


__all__: list[str] = []
