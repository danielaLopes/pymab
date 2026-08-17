from __future__ import annotations

import numpy as np
import pytest

from pymab.benchmarking import (
    BenchmarkResult,
    bootstrap_mean_interval,
    compare,
    standard_error,
)
from pymab.environments import BanditEnvironment
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.simulation import ExperimentConfig


def _benchmark(**kwargs) -> BenchmarkResult:
    return compare(
        {"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)},
        environment=BanditEnvironment(means=np.array([0.0, 0.2, 1.0])),
        config=ExperimentConfig(horizon=40, n_replicates=20, seed=7),
        bootstrap_resamples=200,
        analysis_seed=11,
        baseline="random",
        **kwargs,
    )


def test_benchmark_summary_and_paired_comparison() -> None:
    benchmark = _benchmark()
    summary = benchmark.summary()
    assert len(summary) == 2
    assert benchmark.lowest_mean_regret_policy == "ucb"
    assert summary[0]["n_replicates"] == 20
    assert summary[0]["cumulative_regret_ci_lower"] is not None
    paired = benchmark.compare_to_baseline()
    assert len(paired) == 1
    assert paired[0]["policy_id"] == "ucb"
    assert paired[0]["mean_cumulative_regret_difference"] < 0
    payload = benchmark.to_dict()
    assert payload["lowest_mean_regret_policy"] == "ucb"
    assert payload["paired_comparisons"] == paired


def test_benchmark_analysis_is_deterministic() -> None:
    left = _benchmark().summary()
    right = _benchmark().summary()
    assert left == right


def test_to_dict_computes_summary_once(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _benchmark()
    original = BenchmarkResult.summary
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(BenchmarkResult, "summary", counted)
    benchmark.to_dict()
    assert calls == 1


def test_compare_to_baseline_validation() -> None:
    benchmark = _benchmark()
    with pytest.raises(ValueError, match="baseline"):
        benchmark.compare_to_baseline("missing")
    no_baseline = BenchmarkResult(result=benchmark.result, bootstrap_resamples=10)
    with pytest.raises(ValueError, match="required"):
        no_baseline.compare_to_baseline()
    assert no_baseline.to_dict()["paired_comparisons"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"bootstrap_resamples": 0},
        {"bootstrap_max_index_elements": 0},
        {"bootstrap_resamples": 20, "bootstrap_max_index_elements": 10},
        {"analysis_seed": True},
        {"baseline": "missing"},
    ],
)
def test_benchmark_validation(kwargs) -> None:
    benchmark = _benchmark()
    with pytest.raises((TypeError, ValueError)):
        BenchmarkResult(result=benchmark.result, **kwargs)


def test_bootstrap_interval_and_standard_error() -> None:
    values = np.array([1.0, 2.0, 3.0])
    left = bootstrap_mean_interval(values, n_resamples=100, seed=2)
    right = bootstrap_mean_interval(values, n_resamples=100, seed=2)
    assert left == right
    assert left[0] <= np.mean(values) <= left[1]
    assert standard_error(values) is not None
    assert bootstrap_mean_interval(np.array([1.0])) == (None, None)
    assert standard_error(np.array([1.0])) is None


@pytest.mark.parametrize(
    ("function", "values", "kwargs"),
    [
        (bootstrap_mean_interval, np.ones((2, 2)), {}),
        (bootstrap_mean_interval, np.array([np.nan]), {}),
        (bootstrap_mean_interval, np.ones(2), {"confidence_level": 1}),
        (bootstrap_mean_interval, np.ones(2), {"n_resamples": 0}),
        (bootstrap_mean_interval, np.ones(2), {"max_index_elements": 0}),
        (standard_error, np.array([]), {}),
    ],
)
def test_inference_validation(function, values, kwargs) -> None:
    with pytest.raises(ValueError):
        function(values, **kwargs)


@pytest.mark.optional
def test_benchmark_to_pandas() -> None:
    frame = _benchmark().to_pandas()
    assert list(frame["policy_id"]) == ["random", "ucb"]
