"""Internal deterministic, memory-bounded bootstrap implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pymab.errors import ValidationError
from pymab.statistics import (
    BootstrapConfig,
    ConfidenceMethod,
    IntervalEstimate,
    ResamplingUnit,
)
from pymab.types import FloatArray, IntArray
from pymab.validation import float_array


def summarize_observations(
    contributions: FloatArray,
    *,
    config: BootstrapConfig,
    resampling_unit: ResamplingUnit,
    weights: FloatArray | None = None,
    clusters: Sequence[str | int] | None = None,
) -> IntervalEstimate:
    """Summarize mean or ratio contributions using the requested sampling unit."""

    values = float_array(contributions, name="contributions", ndim=1)
    denominators = _validated_weights(weights, shape=values.shape)
    if resampling_unit is ResamplingUnit.CLUSTER:
        if clusters is None:
            raise ValidationError("cluster resampling requires cluster identifiers")
        numerators, aggregate_denominators = _cluster_aggregates(
            contributions=values,
            weights=denominators,
            clusters=clusters,
        )
    else:
        if clusters is not None:
            raise ValidationError("cluster identifiers require cluster resampling")
        numerators = values
        aggregate_denominators = (
            np.ones(values.size, dtype=float) if denominators is None else denominators
        )
    estimate = _ratio(numerators, aggregate_denominators)
    independent_units = numerators.size
    if independent_units < 2:
        return _without_uncertainty(
            estimate=estimate,
            config=config,
            n_observations=values.size,
            resampling_unit=resampling_unit,
        )
    estimates = _bootstrap_ratios(
        numerators=numerators,
        denominators=aggregate_denominators,
        config=config,
    )
    return _summarize_estimates(
        estimate=estimate,
        estimates=estimates,
        config=config,
        n_observations=values.size,
        resampling_unit=resampling_unit,
    )


def bootstrap_curve(
    replicate_values: object,
    *,
    config: BootstrapConfig,
) -> tuple[FloatArray, FloatArray]:
    """Bootstrap mean curves while sharing replicate draws across outputs."""

    values = float_array(replicate_values, name="replicate_values", ndim=3)
    n_replicates, n_steps, n_policies = values.shape
    if n_replicates < 2:
        mean = np.asarray(np.mean(values, axis=0), dtype=float)
        return mean, mean.copy()
    _require_budget(config, minimum=max(config.n_resamples, n_replicates))
    flattened = values.reshape(n_replicates, n_steps * n_policies)
    outputs_per_chunk = max(1, config.max_chunk_elements // config.n_resamples)
    lower = np.empty(flattened.shape[1], dtype=float)
    upper = np.empty_like(lower)
    alpha = (1.0 - config.confidence_level) / 2.0
    for start in range(0, flattened.shape[1], outputs_per_chunk):
        stop = min(start + outputs_per_chunk, flattened.shape[1])
        rng = np.random.default_rng(config.seed)
        means = np.empty((config.n_resamples, stop - start), dtype=float)
        for resample in range(config.n_resamples):
            indices = rng.integers(0, n_replicates, size=n_replicates)
            means[resample] = np.mean(flattened[indices, start:stop], axis=0)
        quantiles = np.quantile(means, [alpha, 1.0 - alpha], axis=0)
        lower[start:stop] = quantiles[0]
        upper[start:stop] = quantiles[1]
    return lower.reshape(n_steps, n_policies), upper.reshape(n_steps, n_policies)


def _validated_weights(
    weights: FloatArray | None, *, shape: tuple[int, ...]
) -> FloatArray | None:
    if weights is None:
        return None
    result = float_array(weights, name="weights", ndim=1)
    if result.shape != shape:
        raise ValidationError("weights must match contributions shape")
    if np.any(result < 0):
        raise ValidationError("weights must be non-negative")
    if float(np.sum(result)) <= 0:
        raise ValidationError("weights must have a positive total")
    return result


def _cluster_aggregates(
    *,
    contributions: FloatArray,
    weights: FloatArray | None,
    clusters: Sequence[str | int],
) -> tuple[FloatArray, FloatArray]:
    if len(clusters) != contributions.size:
        raise ValidationError("clusters must contain one identifier per observation")
    order = tuple(dict.fromkeys(clusters))
    positions = {cluster: index for index, cluster in enumerate(order)}
    codes: IntArray = np.fromiter(
        (positions[cluster] for cluster in clusters),
        dtype=np.int64,
        count=contributions.size,
    )
    numerators = np.zeros(len(order), dtype=float)
    np.add.at(numerators, codes, contributions)
    denominators = np.zeros(len(order), dtype=float)
    if weights is None:
        np.add.at(denominators, codes, 1.0)
    else:
        np.add.at(denominators, codes, weights)
    return numerators, denominators


def _bootstrap_ratios(
    *,
    numerators: FloatArray,
    denominators: FloatArray,
    config: BootstrapConfig,
) -> FloatArray:
    n_units = numerators.size
    _require_budget(config, minimum=max(config.n_resamples, n_units))
    chunk_size = max(1, config.max_chunk_elements // n_units)
    estimates = np.empty(config.n_resamples, dtype=float)
    rng = np.random.default_rng(config.seed)
    for start in range(0, config.n_resamples, chunk_size):
        stop = min(start + chunk_size, config.n_resamples)
        indices = rng.integers(0, n_units, size=(stop - start, n_units))
        sampled_numerators = np.sum(numerators[indices], axis=1)
        sampled_denominators = np.sum(denominators[indices], axis=1)
        estimates[start:stop] = np.divide(
            sampled_numerators,
            sampled_denominators,
            out=np.full(stop - start, np.nan),
            where=sampled_denominators > 0,
        )
    return estimates


def _summarize_estimates(
    *,
    estimate: float,
    estimates: FloatArray,
    config: BootstrapConfig,
    n_observations: int,
    resampling_unit: ResamplingUnit,
) -> IntervalEstimate:
    finite = estimates[np.isfinite(estimates)]
    if finite.size < 2:
        return _without_uncertainty(
            estimate=estimate,
            config=config,
            n_observations=n_observations,
            resampling_unit=resampling_unit,
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=estimate,
        standard_error=float(np.std(finite, ddof=1)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        confidence_level=config.confidence_level,
        confidence_method=ConfidenceMethod.PERCENTILE_BOOTSTRAP,
        n_observations=n_observations,
        resampling_unit=resampling_unit,
    )


def _without_uncertainty(
    *,
    estimate: float,
    config: BootstrapConfig,
    n_observations: int,
    resampling_unit: ResamplingUnit,
) -> IntervalEstimate:
    return IntervalEstimate(
        estimate=estimate,
        standard_error=None,
        ci_lower=None,
        ci_upper=None,
        confidence_level=config.confidence_level,
        confidence_method=ConfidenceMethod.PERCENTILE_BOOTSTRAP,
        n_observations=n_observations,
        resampling_unit=resampling_unit,
    )


def _ratio(numerators: FloatArray, denominators: FloatArray) -> float:
    denominator = float(np.sum(denominators))
    if denominator <= 0:
        raise ValidationError("the resampling statistic has a zero denominator")
    result = float(np.sum(numerators) / denominator)
    if not np.isfinite(result):
        raise ValidationError("the resampling statistic must be finite")
    return result


def _require_budget(config: BootstrapConfig, *, minimum: int) -> None:
    if config.max_chunk_elements < minimum:
        raise ValidationError(
            "max_chunk_elements must accommodate one complete resampling unit "
            f"and all {config.n_resamples} scalar estimates; minimum {minimum}"
        )


__all__: list[str] = []
