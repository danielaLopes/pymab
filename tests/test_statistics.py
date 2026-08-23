from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import pymab._resampling as resampling
from pymab._resampling import bootstrap_curve, summarize_observations
from pymab.statistics import (
    BootstrapConfig,
    ConfidenceMethod,
    IntervalEstimate,
    ResamplingUnit,
    bootstrap_mean_interval,
    standard_error,
)


def test_bootstrap_config_is_normalized_and_serializable() -> None:
    config = BootstrapConfig(
        confidence_level=np.float64(0.9),
        n_resamples=np.int64(20),
        seed=np.int64(7),
        max_chunk_elements=np.int64(40),
    )
    assert config == BootstrapConfig(0.9, 20, 7, 40)
    assert config.to_dict() == {
        "confidence_level": 0.9,
        "n_resamples": 20,
        "seed": 7,
        "max_chunk_elements": 40,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_level", 0),
        ("confidence_level", 1),
        ("confidence_level", np.nan),
        ("confidence_level", True),
        ("n_resamples", 0),
        ("n_resamples", True),
        ("seed", True),
        ("seed", 1.5),
        ("max_chunk_elements", 0),
        ("max_chunk_elements", True),
    ],
)
def test_bootstrap_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BootstrapConfig(**{field: value})


def test_bootstrap_config_rejects_impossible_budget() -> None:
    with pytest.raises(ValueError, match="at least n_resamples"):
        BootstrapConfig(n_resamples=20, max_chunk_elements=19)


def test_interval_estimate_validation_and_serialization() -> None:
    estimate = IntervalEstimate(
        estimate=2,
        standard_error=0.5,
        ci_lower=1,
        ci_upper=3,
        confidence_level=0.95,
        confidence_method=ConfidenceMethod.PERCENTILE_BOOTSTRAP,
        n_observations=4,
        resampling_unit=ResamplingUnit.EVENT,
    )
    assert estimate.to_dict()["confidence_method"] == "percentile_bootstrap"
    assert estimate.to_dict()["resampling_unit"] == "event"
    with pytest.raises(ValueError, match="both"):
        replace(estimate, ci_upper=None)
    with pytest.raises(ValueError, match="non-negative"):
        replace(estimate, standard_error=-1)
    with pytest.raises(ValueError, match="exceed"):
        replace(estimate, ci_lower=4)
    with pytest.raises(ValueError, match="confidence_level"):
        replace(estimate, confidence_level=1)


def test_mean_interval_matches_fixed_seed_reference_draws() -> None:
    values = np.array([1.0, 2.0, 5.0, 8.0])
    config = BootstrapConfig(n_resamples=50, seed=12, max_chunk_elements=1_000)
    estimate = bootstrap_mean_interval(values, config=config)
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, values.size, size=(config.n_resamples, values.size))
    draws = np.mean(values[indices], axis=1)
    alpha = (1.0 - config.confidence_level) / 2.0
    expected_lower, expected_upper = np.quantile(draws, [alpha, 1.0 - alpha])
    assert estimate.estimate == np.mean(values)
    assert estimate.standard_error == np.std(draws, ddof=1)
    assert estimate.ci_lower == expected_lower
    assert estimate.ci_upper == expected_upper
    assert estimate.resampling_unit is ResamplingUnit.REPLICATE


def test_scalar_bootstrap_is_bitwise_chunk_invariant() -> None:
    values = np.array([1.0, 2.0, 5.0, 8.0])
    compact = BootstrapConfig(n_resamples=80, seed=3, max_chunk_elements=80)
    roomy = replace(compact, max_chunk_elements=8_000)
    assert bootstrap_mean_interval(values, config=compact) == bootstrap_mean_interval(
        values, config=roomy
    )


def test_ratio_bootstrap_matches_fixed_seed_reference_draws() -> None:
    contributions = np.array([0.0, 2.0, 6.0])
    weights = np.array([0.0, 1.0, 2.0])
    config = BootstrapConfig(n_resamples=60, seed=4, max_chunk_elements=600)
    estimate = summarize_observations(
        contributions,
        weights=weights,
        config=config,
        resampling_unit=ResamplingUnit.EVENT,
    )
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, 3, size=(config.n_resamples, 3))
    numerator = np.sum(contributions[indices], axis=1)
    denominator = np.sum(weights[indices], axis=1)
    draws = numerator[denominator > 0] / denominator[denominator > 0]
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    assert estimate.estimate == pytest.approx(8 / 3)
    assert estimate.standard_error == np.std(draws, ddof=1)
    assert estimate.ci_lower == lower
    assert estimate.ci_upper == upper


def test_unequal_cluster_bootstrap_uses_cluster_sums_and_event_counts() -> None:
    contributions = np.array([1.0, 3.0, 8.0, 10.0, 12.0])
    clusters = ("small", "large", "large", "large", "large")
    config = BootstrapConfig(n_resamples=40, seed=9, max_chunk_elements=400)
    estimate = summarize_observations(
        contributions,
        clusters=clusters,
        config=config,
        resampling_unit=ResamplingUnit.CLUSTER,
    )
    cluster_sums = np.array([1.0, 33.0])
    cluster_counts = np.array([1.0, 4.0])
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, 2, size=(config.n_resamples, 2))
    draws = np.sum(cluster_sums[indices], axis=1) / np.sum(
        cluster_counts[indices], axis=1
    )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    assert estimate.estimate == np.mean(contributions)
    assert estimate.ci_lower == lower
    assert estimate.ci_upper == upper
    assert estimate.resampling_unit is ResamplingUnit.CLUSTER


def test_one_independent_unit_returns_point_estimate_only() -> None:
    estimate = summarize_observations(
        np.array([1.0, 3.0]),
        clusters=("only", "only"),
        config=BootstrapConfig(n_resamples=10),
        resampling_unit=ResamplingUnit.CLUSTER,
    )
    assert estimate.estimate == 2
    assert estimate.standard_error is None
    assert estimate.ci_lower is None
    assert estimate.ci_upper is None


def test_resampling_contract_validation() -> None:
    config = BootstrapConfig(n_resamples=10)
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="requires cluster"):
        summarize_observations(
            values,
            config=config,
            resampling_unit=ResamplingUnit.CLUSTER,
        )
    with pytest.raises(ValueError, match="require cluster resampling"):
        summarize_observations(
            values,
            clusters=("a", "b"),
            config=config,
            resampling_unit=ResamplingUnit.EVENT,
        )
    with pytest.raises(ValueError, match="match"):
        summarize_observations(
            values,
            weights=np.array([1.0]),
            config=config,
            resampling_unit=ResamplingUnit.EVENT,
        )
    with pytest.raises(ValueError, match="non-negative"):
        summarize_observations(
            values,
            weights=np.array([1.0, -1.0]),
            config=config,
            resampling_unit=ResamplingUnit.EVENT,
        )
    with pytest.raises(ValueError, match="positive total"):
        summarize_observations(
            values,
            weights=np.zeros(2),
            config=config,
            resampling_unit=ResamplingUnit.EVENT,
        )
    with pytest.raises(ValueError, match="one identifier"):
        summarize_observations(
            values,
            clusters=("a",),
            config=config,
            resampling_unit=ResamplingUnit.CLUSTER,
        )


def test_cluster_ratio_bootstrap_is_chunk_invariant() -> None:
    values = np.array([1.0, 2.0, 3.0, 8.0])
    weights = np.array([1.0, 1.0, 2.0, 4.0])
    clusters = ("a", "a", "b", "c")
    compact = BootstrapConfig(n_resamples=20, seed=9, max_chunk_elements=20)
    roomy = replace(compact, max_chunk_elements=2_000)
    left = summarize_observations(
        values,
        weights=weights,
        clusters=clusters,
        config=compact,
        resampling_unit=ResamplingUnit.CLUSTER,
    )
    right = summarize_observations(
        values,
        weights=weights,
        clusters=clusters,
        config=roomy,
        resampling_unit=ResamplingUnit.CLUSTER,
    )
    assert left == right


def test_unavailable_resample_statistics_return_no_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resampling,
        "_bootstrap_ratios",
        lambda **kwargs: np.array([np.nan, np.nan]),
    )
    estimate = summarize_observations(
        np.array([1.0, 2.0]),
        config=BootstrapConfig(n_resamples=2),
        resampling_unit=ResamplingUnit.EVENT,
    )
    assert estimate.standard_error is None


def test_internal_ratio_rejects_invalid_aggregate_statistics() -> None:
    with pytest.raises(ValueError, match="zero denominator"):
        resampling._ratio(np.array([1.0]), np.array([0.0]))
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        resampling._ratio(
            np.array([np.finfo(float).max, np.finfo(float).max]),
            np.ones(2),
        )


def test_curve_bootstrap_matches_reference_and_is_chunk_invariant() -> None:
    values = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    roomy = BootstrapConfig(n_resamples=30, seed=5, max_chunk_elements=300)
    compact = replace(roomy, max_chunk_elements=30)
    lower, upper = bootstrap_curve(values, config=roomy)
    chunked_lower, chunked_upper = bootstrap_curve(values, config=compact)
    rng = np.random.default_rng(roomy.seed)
    indices = rng.integers(0, 4, size=(roomy.n_resamples, 4))
    draws = np.mean(values[indices], axis=1)
    alpha = (1.0 - roomy.confidence_level) / 2.0
    expected = np.quantile(draws, [alpha, 1.0 - alpha], axis=0)
    np.testing.assert_array_equal(lower, expected[0])
    np.testing.assert_array_equal(upper, expected[1])
    np.testing.assert_array_equal(chunked_lower, lower)
    np.testing.assert_array_equal(chunked_upper, upper)


def test_resampling_budget_rejects_an_indivisible_unit() -> None:
    config = BootstrapConfig(n_resamples=2, max_chunk_elements=2)
    with pytest.raises(ValueError, match="minimum 3"):
        bootstrap_mean_interval(np.array([1.0, 2.0, 3.0]), config=config)
    with pytest.raises(ValueError, match="minimum 3"):
        bootstrap_curve(np.ones((3, 1, 1)), config=config)


def test_requested_empty_workspaces_respect_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 40
    requested: list[int] = []
    original_empty = resampling.np.empty

    def tracked_empty(shape, *args, **kwargs):
        requested.append(int(np.prod(shape)) if not isinstance(shape, int) else shape)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(resampling.np, "empty", tracked_empty)
    bootstrap_curve(
        np.arange(4 * 5 * 3, dtype=float).reshape(4, 5, 3),
        config=BootstrapConfig(
            n_resamples=40,
            seed=1,
            max_chunk_elements=budget,
        ),
    )
    assert requested
    assert max(requested) <= budget


def test_standard_error_validates_and_handles_one_observation() -> None:
    assert standard_error([1.0]) is None
    assert standard_error([1.0, 3.0]) == 1.0
    with pytest.raises(ValueError):
        standard_error([])
    with pytest.raises(TypeError, match="BootstrapConfig"):
        bootstrap_mean_interval([1.0, 2.0], config=object())  # type: ignore[arg-type]
