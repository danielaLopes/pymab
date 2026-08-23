from __future__ import annotations

import numpy as np
import pytest

from pymab.benchmarking import (
    BenchmarkConfig,
    BenchmarkResult,
    PolicyComparison,
    PolicySummary,
    compare,
)
from pymab.distributions import GaussianReward
from pymab.environments import BanditEnvironment
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.policies.policy import Policy
from pymab.simulation import ExperimentConfig
from pymab.statistics import BootstrapConfig


class _FixedActionPolicy(Policy):
    def __init__(self, action: int) -> None:
        self.action = action
        super().__init__(n_arms=2)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return self.action

    def update(self, *, action: int, reward: float) -> None:
        return None

    def reset(self) -> None:
        return None

    def recommend_action(self) -> int:
        return self.action


def _benchmark(*, analysis: BenchmarkConfig | None = None) -> BenchmarkResult:
    return compare(
        {"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)},
        environment=BanditEnvironment(means=np.array([0.0, 0.2, 1.0])),
        config=ExperimentConfig(horizon=40, n_replicates=20, seed=7),
        analysis=(
            BenchmarkConfig(
                bootstrap=BootstrapConfig(n_resamples=200, seed=11),
                baseline="random",
            )
            if analysis is None
            else analysis
        ),
    )


def test_benchmark_summary_and_paired_comparison() -> None:
    benchmark = _benchmark()
    summary = benchmark.summary()
    assert len(summary) == 2
    assert all(isinstance(record, PolicySummary) for record in summary)
    assert benchmark.lowest_mean_regret_policy == "ucb"
    assert summary[0].n_replicates == 20
    assert summary[0].cumulative_regret.ci_lower is not None
    paired = benchmark.compare_to_baseline()
    assert len(paired) == 1
    assert isinstance(paired[0], PolicyComparison)
    assert paired[0].policy_id == "ucb"
    assert paired[0].cumulative_regret.estimate < 0
    payload = benchmark.to_dict()
    assert payload["lowest_mean_regret_policy"] == "ucb"
    assert payload["paired_comparisons"] == [paired[0].to_dict()]


def test_benchmark_analysis_is_deterministic() -> None:
    left = _benchmark().summary()
    right = _benchmark().summary()
    assert left == right


def test_paired_comparison_has_reference_monte_carlo_coverage() -> None:
    estimates: list[float] = []
    covered = 0
    for trial in range(20):
        benchmark = compare(
            {"low": _FixedActionPolicy(0), "high": _FixedActionPolicy(1)},
            environment=BanditEnvironment(
                means=np.array([0.0, 1.0]),
                reward_model=GaussianReward(std=1.0),
            ),
            config=ExperimentConfig(horizon=1, n_replicates=80, seed=trial),
            analysis=BenchmarkConfig(
                bootstrap=BootstrapConfig(n_resamples=200, seed=100 + trial),
                baseline="low",
            ),
        )
        interval = benchmark.compare_to_baseline()[0].total_reward
        assert interval.ci_lower is not None
        assert interval.ci_upper is not None
        estimates.append(interval.estimate)
        covered += int(interval.ci_lower <= 1.0 <= interval.ci_upper)
    assert np.mean(estimates) == pytest.approx(1.0, abs=0.05)
    assert covered >= 17


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
    no_baseline = BenchmarkResult(
        result=benchmark.result,
        config=BenchmarkConfig(bootstrap=BootstrapConfig(n_resamples=20)),
    )
    with pytest.raises(ValueError, match="required"):
        no_baseline.compare_to_baseline()
    assert no_baseline.to_dict()["paired_comparisons"] == []


@pytest.mark.parametrize(
    "config",
    [
        BenchmarkConfig(
            bootstrap=BootstrapConfig(n_resamples=20, max_chunk_elements=20),
            baseline="missing",
        ),
        BenchmarkConfig(
            bootstrap=BootstrapConfig(n_resamples=10, max_chunk_elements=10)
        ),
    ],
)
def test_benchmark_validation(config: BenchmarkConfig) -> None:
    benchmark = _benchmark()
    with pytest.raises(ValueError):
        BenchmarkResult(result=benchmark.result, config=config)


def test_benchmark_configuration_validation() -> None:
    with pytest.raises(TypeError, match="BootstrapConfig"):
        BenchmarkConfig(bootstrap=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        BenchmarkConfig(baseline="")
    with pytest.raises(TypeError, match="SimulationResult"):
        BenchmarkResult(result=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BenchmarkConfig"):
        BenchmarkResult(
            result=_benchmark().result,
            config=object(),  # type: ignore[arg-type]
        )


@pytest.mark.optional
def test_benchmark_to_pandas() -> None:
    frame = _benchmark().to_pandas()
    assert list(frame["policy_id"]) == ["random", "ucb"]
