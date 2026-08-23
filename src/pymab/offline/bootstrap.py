"""Memory-bounded bootstrap services for offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymab.offline.data import EstimateMethod, ResamplingUnit
from pymab.types import FloatArray


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap uncertainty summary."""

    standard_error: float | None
    lower: float | None
    upper: float | None
    unit: ResamplingUnit


@dataclass(frozen=True)
class Bootstrapper:
    """Deterministic event or cluster percentile bootstrap."""

    n_resamples: int
    confidence_level: float
    seed: int
    max_index_elements: int = 1_000_000

    def summarize(
        self,
        *,
        contributions: FloatArray,
        weights: FloatArray,
        method: EstimateMethod,
        clusters: tuple[str | int, ...] | None,
    ) -> BootstrapResult:
        unit = ResamplingUnit.CLUSTER if clusters is not None else ResamplingUnit.EVENT
        independent_units = (
            len(set(clusters)) if clusters is not None else contributions.size
        )
        if independent_units < 2:
            return BootstrapResult(None, None, None, unit)
        estimates = (
            self._cluster_estimates(
                contributions=contributions,
                weights=weights,
                method=method,
                clusters=clusters,
            )
            if clusters is not None
            else self._event_estimates(
                contributions=contributions,
                weights=weights,
                method=method,
            )
        )
        finite = estimates[np.isfinite(estimates)]
        if finite.size < 2:
            return BootstrapResult(None, None, None, unit)
        alpha = (1.0 - self.confidence_level) / 2.0
        lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        return BootstrapResult(
            standard_error=float(np.std(finite, ddof=1)),
            lower=float(lower),
            upper=float(upper),
            unit=unit,
        )

    def _event_estimates(
        self,
        *,
        contributions: FloatArray,
        weights: FloatArray,
        method: EstimateMethod,
    ) -> FloatArray:
        rng = np.random.default_rng(self.seed)
        estimates = np.empty(self.n_resamples, dtype=float)
        chunk_size = max(1, self.max_index_elements // contributions.size)
        for start in range(0, self.n_resamples, chunk_size):
            stop = min(start + chunk_size, self.n_resamples)
            indices = rng.integers(
                0,
                contributions.size,
                size=(stop - start, contributions.size),
            )
            estimates[start:stop] = _estimates_from_indices(
                contributions=contributions,
                weights=weights,
                method=method,
                indices=indices,
            )
        return estimates

    def _cluster_estimates(
        self,
        *,
        contributions: FloatArray,
        weights: FloatArray,
        method: EstimateMethod,
        clusters: tuple[str | int, ...],
    ) -> FloatArray:
        cluster_order = tuple(dict.fromkeys(clusters))
        cluster_values = np.asarray(clusters, dtype=object)
        rows = {
            cluster: np.flatnonzero(cluster_values == cluster)
            for cluster in cluster_order
        }
        rng = np.random.default_rng(self.seed)
        estimates = np.empty(self.n_resamples, dtype=float)
        for resample in range(self.n_resamples):
            selected = rng.integers(0, len(cluster_order), size=len(cluster_order))
            indices = np.concatenate([rows[cluster_order[index]] for index in selected])
            sampled_contributions = contributions[indices]
            if method is EstimateMethod.SNIPS:
                denominator = float(np.sum(weights[indices]))
                estimates[resample] = (
                    np.nan
                    if denominator <= 0
                    else float(np.sum(sampled_contributions) / denominator)
                )
            else:
                estimates[resample] = float(np.mean(sampled_contributions))
        return estimates


def _estimates_from_indices(
    *,
    contributions: FloatArray,
    weights: FloatArray,
    method: EstimateMethod,
    indices: np.ndarray,
) -> FloatArray:
    sampled_contributions = contributions[indices]
    if method is not EstimateMethod.SNIPS:
        return np.asarray(np.mean(sampled_contributions, axis=1), dtype=float)
    denominators = np.sum(weights[indices], axis=1)
    numerators = np.sum(sampled_contributions, axis=1)
    return np.asarray(
        np.divide(
            numerators,
            denominators,
            out=np.full(indices.shape[0], np.nan),
            where=denominators > 0,
        ),
        dtype=float,
    )


__all__ = ["BootstrapResult", "Bootstrapper"]
