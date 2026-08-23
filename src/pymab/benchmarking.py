"""Paired, replicate-aware policy benchmarking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np

from pymab.environments import Environment
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult

SummaryValue: TypeAlias = str | int | float | None
SummaryRow: TypeAlias = dict[str, SummaryValue]


@dataclass(frozen=True)
class _MetricEstimate:
    mean: float
    standard_error: float | None
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class _PolicySummaryRecord:
    policy_index: int
    policy_id: str
    n_replicates: int
    metrics: tuple[tuple[str, _MetricEstimate], ...]

    def to_dict(self) -> SummaryRow:
        row: SummaryRow = {
            "policy_index": self.policy_index,
            "policy_id": self.policy_id,
            "n_replicates": self.n_replicates,
        }
        for name, estimate in self.metrics:
            row[f"mean_{name}"] = estimate.mean
            row[f"{name}_standard_error"] = estimate.standard_error
            row[f"{name}_ci_lower"] = estimate.lower
            row[f"{name}_ci_upper"] = estimate.upper
        return row


@dataclass(frozen=True)
class _PolicyComparisonRecord:
    policy_id: str
    baseline_id: str
    n_replicates: int
    metrics: tuple[tuple[str, _MetricEstimate], ...]

    def to_dict(self) -> SummaryRow:
        row: SummaryRow = {
            "policy_id": self.policy_id,
            "baseline_id": self.baseline_id,
            "n_replicates": self.n_replicates,
        }
        for name, estimate in self.metrics:
            row[f"mean_{name}_difference"] = estimate.mean
            row[f"{name}_difference_standard_error"] = estimate.standard_error
            row[f"{name}_difference_ci_lower"] = estimate.lower
            row[f"{name}_difference_ci_upper"] = estimate.upper
        return row


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """A simulation plus deterministic replicate-level uncertainty analysis."""

    result: SimulationResult
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_max_index_elements: int = 1_000_000
    analysis_seed: int = 0
    baseline: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not isinstance(self.bootstrap_resamples, int) or isinstance(
            self.bootstrap_resamples, bool
        ):
            raise TypeError("bootstrap_resamples must be an integer")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not isinstance(self.bootstrap_max_index_elements, int) or isinstance(
            self.bootstrap_max_index_elements, bool
        ):
            raise TypeError("bootstrap_max_index_elements must be an integer")
        if self.bootstrap_max_index_elements <= 0:
            raise ValueError("bootstrap_max_index_elements must be positive")
        minimum_budget = max(self.bootstrap_resamples, self.result.n_replicates)
        if self.bootstrap_max_index_elements < minimum_budget:
            raise ValueError(
                "bootstrap_max_index_elements must accommodate at least one "
                "bootstrap unit and one plot-band column"
            )
        if self.baseline is not None and self.baseline not in self.result.policy_ids:
            raise ValueError("baseline must identify a policy in the result")
        if isinstance(self.analysis_seed, bool) or not isinstance(
            self.analysis_seed, Integral
        ):
            raise TypeError("analysis_seed must be an integer")
        object.__setattr__(self, "analysis_seed", int(self.analysis_seed))

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return self.result.policy_ids

    @property
    def lowest_mean_regret_policy(self) -> str:
        """Policy with the lowest point estimate of final cumulative regret."""

        rows = self.summary()
        return str(
            min(
                rows,
                key=lambda row: cast(float, row["mean_cumulative_regret"]),
            )["policy_id"]
        )

    def summary(self) -> list[SummaryRow]:
        """Return point estimates and bootstrap intervals by policy."""

        cumulative_regret = self.result.cumulative_regret_by_replicate[:, -1, :]
        total_reward = np.sum(self.result.rewards, axis=1)
        optimal_rate = np.mean(self.result.optimal_action_indicator, axis=1)
        final_simple_regret = self.result.simple_regret[:, -1, :]
        records: list[_PolicySummaryRecord] = []
        for index, policy_id in enumerate(self.policy_ids):
            metrics = {
                "cumulative_regret": cumulative_regret[:, index],
                "total_reward": total_reward[:, index],
                "optimal_action_rate": optimal_rate[:, index],
                "final_simple_regret": final_simple_regret[:, index],
            }
            records.append(
                _PolicySummaryRecord(
                    policy_index=index,
                    policy_id=policy_id,
                    n_replicates=self.result.n_replicates,
                    metrics=tuple(
                        (
                            metric,
                            _summarize_metric(
                                values,
                                confidence_level=self.confidence_level,
                                n_resamples=self.bootstrap_resamples,
                                max_index_elements=self.bootstrap_max_index_elements,
                                seed=_metric_seed(
                                    self.analysis_seed, policy_id, metric
                                ),
                            ),
                        )
                        for metric, values in metrics.items()
                    ),
                )
            )
        return [record.to_dict() for record in records]

    def compare_to_baseline(self, baseline: str | None = None) -> list[SummaryRow]:
        """Return paired replicate differences relative to a baseline policy."""

        baseline_id = self.baseline if baseline is None else baseline
        if baseline_id is None:
            raise ValueError("a baseline policy ID is required")
        if baseline_id not in self.policy_ids:
            raise ValueError("baseline must identify a policy in the result")
        baseline_index = self.policy_ids.index(baseline_id)
        metrics = {
            "cumulative_regret": self.result.cumulative_regret_by_replicate[:, -1, :],
            "total_reward": np.sum(self.result.rewards, axis=1),
            "optimal_action_rate": np.mean(
                self.result.optimal_action_indicator, axis=1
            ),
            "final_simple_regret": self.result.simple_regret[:, -1, :],
        }
        records: list[_PolicyComparisonRecord] = []
        for index, policy_id in enumerate(self.policy_ids):
            if policy_id == baseline_id:
                continue
            records.append(
                _PolicyComparisonRecord(
                    policy_id=policy_id,
                    baseline_id=baseline_id,
                    n_replicates=self.result.n_replicates,
                    metrics=tuple(
                        (
                            metric,
                            _summarize_metric(
                                values[:, index] - values[:, baseline_index],
                                confidence_level=self.confidence_level,
                                n_resamples=self.bootstrap_resamples,
                                max_index_elements=self.bootstrap_max_index_elements,
                                seed=_metric_seed(
                                    self.analysis_seed,
                                    policy_id,
                                    baseline_id,
                                    metric,
                                ),
                            ),
                        )
                        for metric, values in metrics.items()
                    ),
                )
            )
        return [record.to_dict() for record in records]

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pymab[analysis] to use to_pandas().") from exc
        return pd.DataFrame.from_records(self.summary())

    def to_dict(self) -> dict[str, Any]:
        summary = self.summary()
        lowest = str(
            min(
                summary,
                key=lambda row: cast(float, row["mean_cumulative_regret"]),
            )["policy_id"]
        )
        return {
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_max_index_elements": self.bootstrap_max_index_elements,
            "analysis_seed": self.analysis_seed,
            "baseline": self.baseline,
            "lowest_mean_regret_policy": lowest,
            "summary": summary,
            "paired_comparisons": (
                [] if self.baseline is None else self.compare_to_baseline()
            ),
            "result": self.result.to_dict(),
        }

    def plot_average_reward(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        from pymab.plotting import BootstrapBandConfig, plot_average_reward

        return plot_average_reward(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=BootstrapBandConfig(
                confidence_level=self.confidence_level,
                n_resamples=self.bootstrap_resamples,
                seed=self.analysis_seed,
                max_output_elements=self.bootstrap_max_index_elements,
            ),
        )

    def plot_cumulative_regret(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        from pymab.plotting import BootstrapBandConfig, plot_cumulative_regret

        return plot_cumulative_regret(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=BootstrapBandConfig(
                confidence_level=self.confidence_level,
                n_resamples=self.bootstrap_resamples,
                seed=self.analysis_seed,
                max_output_elements=self.bootstrap_max_index_elements,
            ),
        )

    def plot_optimal_action_rate(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        from pymab.plotting import BootstrapBandConfig, plot_optimal_action_rate

        return plot_optimal_action_rate(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=BootstrapBandConfig(
                confidence_level=self.confidence_level,
                n_resamples=self.bootstrap_resamples,
                seed=self.analysis_seed,
                max_output_elements=self.bootstrap_max_index_elements,
            ),
        )


def _summarize_metric(
    values: np.ndarray,
    *,
    confidence_level: float,
    n_resamples: int,
    max_index_elements: int,
    seed: int,
) -> _MetricEstimate:
    lower, upper = bootstrap_mean_interval(
        values,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        max_index_elements=max_index_elements,
        seed=seed,
    )
    return _MetricEstimate(
        mean=float(np.mean(values)),
        standard_error=standard_error(values),
        lower=lower,
        upper=upper,
    )


def compare(
    policies: Mapping[str, Policy | ContextualPolicy],
    *,
    environment: Environment,
    config: ExperimentConfig,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_max_index_elements: int = 1_000_000,
    analysis_seed: int = 0,
    baseline: str | None = None,
) -> BenchmarkResult:
    """Run one paired experiment and summarize independent replicates."""

    result = Experiment(
        environment=environment,
        policies=policies,
        config=config,
    ).run()
    return BenchmarkResult(
        result=result,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_max_index_elements=bootstrap_max_index_elements,
        analysis_seed=analysis_seed,
        baseline=baseline,
    )


def standard_error(values: np.ndarray) -> float | None:
    """Standard error across independent replicates."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a non-empty finite 1D array")
    if data.size < 2:
        return None
    return float(np.std(data, ddof=1) / np.sqrt(data.size))


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    max_index_elements: int = 1_000_000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Deterministic percentile-bootstrap interval for a replicate mean."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a non-empty finite 1D array")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not isinstance(max_index_elements, int) or isinstance(max_index_elements, bool):
        raise TypeError("max_index_elements must be an integer")
    if max_index_elements <= 0:
        raise ValueError("max_index_elements must be positive")
    if max_index_elements < data.size:
        raise ValueError(
            "max_index_elements must accommodate one complete bootstrap sample"
        )
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if data.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    chunk_size = max(1, max_index_elements // data.size)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, data.size, size=(stop - start, data.size))
        means[start:stop] = np.mean(data[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _metric_seed(master: int, *parts: str) -> int:
    payload = "\x00".join([str(master), *parts]).encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=b"pymab-ci").digest(),
        byteorder="little",
        signed=False,
    )


__all__ = [
    "BenchmarkResult",
    "bootstrap_mean_interval",
    "compare",
    "standard_error",
]
