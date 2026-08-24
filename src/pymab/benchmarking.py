"""Paired, replicate-aware policy benchmarking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from pymab.environments import Environment
from pymab.errors import ValidationError
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.results import SimulationResult
from pymab.simulation import Experiment, ExperimentConfig
from pymab.statistics import BootstrapConfig, IntervalEstimate, bootstrap_mean_interval

SummaryValue: TypeAlias = str | int | float | None
SummaryRow: TypeAlias = dict[str, SummaryValue]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for replicate-level benchmark analysis."""

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    baseline: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap, BootstrapConfig):
            raise TypeError("bootstrap must be a BootstrapConfig")
        if self.baseline is not None and (
            not isinstance(self.baseline, str) or not self.baseline
        ):
            raise ValidationError("baseline must be a non-empty policy ID")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible analysis configuration."""

        return {"bootstrap": self.bootstrap.to_dict(), "baseline": self.baseline}


@dataclass(frozen=True)
class PolicySummary:
    """Typed uncertainty estimates for one policy's benchmark metrics."""

    policy_index: int
    policy_id: str
    cumulative_regret: IntervalEstimate
    total_reward: IntervalEstimate
    optimal_action_rate: IntervalEstimate
    final_simple_regret: IntervalEstimate

    @property
    def n_replicates(self) -> int:
        """Number of independent simulation replicates in each estimate."""

        return self.cumulative_regret.n_observations

    def to_dict(self) -> SummaryRow:
        """Return a flat JSON-compatible summary row."""

        row: SummaryRow = {
            "policy_index": self.policy_index,
            "policy_id": self.policy_id,
            "n_replicates": self.n_replicates,
        }
        _add_metric(row, "cumulative_regret", self.cumulative_regret)
        _add_metric(row, "total_reward", self.total_reward)
        _add_metric(row, "optimal_action_rate", self.optimal_action_rate)
        _add_metric(row, "final_simple_regret", self.final_simple_regret)
        return row


@dataclass(frozen=True)
class PolicyComparison:
    """Typed paired metric differences relative to a baseline policy."""

    policy_id: str
    baseline_id: str
    cumulative_regret: IntervalEstimate
    total_reward: IntervalEstimate
    optimal_action_rate: IntervalEstimate
    final_simple_regret: IntervalEstimate

    @property
    def n_replicates(self) -> int:
        """Number of paired simulation replicates in each estimate."""

        return self.cumulative_regret.n_observations

    def to_dict(self) -> SummaryRow:
        """Return a flat JSON-compatible comparison row."""

        row: SummaryRow = {
            "policy_id": self.policy_id,
            "baseline_id": self.baseline_id,
            "n_replicates": self.n_replicates,
        }
        _add_metric(row, "cumulative_regret", self.cumulative_regret, difference=True)
        _add_metric(row, "total_reward", self.total_reward, difference=True)
        _add_metric(
            row, "optimal_action_rate", self.optimal_action_rate, difference=True
        )
        _add_metric(
            row, "final_simple_regret", self.final_simple_regret, difference=True
        )
        return row


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """A simulation plus deterministic replicate-level uncertainty analysis."""

    result: SimulationResult
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.result, SimulationResult):
            raise TypeError("result must be a SimulationResult")
        if not isinstance(self.config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig")
        if (
            self.config.baseline is not None
            and self.config.baseline not in self.result.policy_ids
        ):
            raise ValidationError("baseline must identify a policy in the result")
        minimum_budget = max(
            self.config.bootstrap.n_resamples,
            self.result.n_replicates,
        )
        if self.config.bootstrap.max_chunk_elements < minimum_budget:
            raise ValidationError(
                "max_chunk_elements must accommodate the requested resamples and "
                "one complete replicate sample"
            )

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return self.result.policy_ids

    @property
    def lowest_mean_regret_policy(self) -> str:
        """Policy with the lowest final cumulative-regret point estimate."""

        return min(
            self.summary(),
            key=lambda summary: summary.cumulative_regret.estimate,
        ).policy_id

    def summary(self) -> tuple[PolicySummary, ...]:
        """Return typed point estimates and bootstrap intervals by policy."""

        metrics = _benchmark_metrics(self.result)
        return tuple(
            PolicySummary(
                policy_index=index,
                policy_id=policy_id,
                cumulative_regret=self._summarize(
                    metrics["cumulative_regret"][:, index],
                    policy_id,
                    "cumulative_regret",
                ),
                total_reward=self._summarize(
                    metrics["total_reward"][:, index], policy_id, "total_reward"
                ),
                optimal_action_rate=self._summarize(
                    metrics["optimal_action_rate"][:, index],
                    policy_id,
                    "optimal_action_rate",
                ),
                final_simple_regret=self._summarize(
                    metrics["final_simple_regret"][:, index],
                    policy_id,
                    "final_simple_regret",
                ),
            )
            for index, policy_id in enumerate(self.policy_ids)
        )

    def compare_to_baseline(
        self, baseline: str | None = None
    ) -> tuple[PolicyComparison, ...]:
        """Return paired replicate differences relative to a baseline policy."""

        baseline_id = self.config.baseline if baseline is None else baseline
        if baseline_id is None:
            raise ValidationError("a baseline policy ID is required")
        if baseline_id not in self.policy_ids:
            raise ValidationError("baseline must identify a policy in the result")
        baseline_index = self.policy_ids.index(baseline_id)
        metrics = _benchmark_metrics(self.result)
        return tuple(
            PolicyComparison(
                policy_id=policy_id,
                baseline_id=baseline_id,
                cumulative_regret=self._summarize_difference(
                    metrics["cumulative_regret"],
                    index,
                    baseline_index,
                    policy_id,
                    baseline_id,
                    "cumulative_regret",
                ),
                total_reward=self._summarize_difference(
                    metrics["total_reward"],
                    index,
                    baseline_index,
                    policy_id,
                    baseline_id,
                    "total_reward",
                ),
                optimal_action_rate=self._summarize_difference(
                    metrics["optimal_action_rate"],
                    index,
                    baseline_index,
                    policy_id,
                    baseline_id,
                    "optimal_action_rate",
                ),
                final_simple_regret=self._summarize_difference(
                    metrics["final_simple_regret"],
                    index,
                    baseline_index,
                    policy_id,
                    baseline_id,
                    "final_simple_regret",
                ),
            )
            for index, policy_id in enumerate(self.policy_ids)
            if policy_id != baseline_id
        )

    def to_pandas(self) -> Any:
        """Return summary rows as a pandas DataFrame."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pymab[analysis] to use to_pandas().") from exc
        return pd.DataFrame.from_records(record.to_dict() for record in self.summary())

    def to_dict(self) -> dict[str, object]:
        """Return the analysis and simulation as a JSON-compatible record."""

        summaries = self.summary()
        lowest = min(
            summaries,
            key=lambda summary: summary.cumulative_regret.estimate,
        ).policy_id
        return {
            "analysis": self.config.to_dict(),
            "lowest_mean_regret_policy": lowest,
            "summary": [record.to_dict() for record in summaries],
            "paired_comparisons": (
                []
                if self.config.baseline is None
                else [
                    record.to_dict()
                    for record in self.compare_to_baseline(self.config.baseline)
                ]
            ),
            "result": self.result.to_dict(),
        }

    def plot_average_reward(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot replicate-aware average reward curves."""

        from pymab.plotting import plot_average_reward

        return plot_average_reward(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=self.config.bootstrap,
        )

    def plot_cumulative_regret(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot replicate-aware cumulative regret curves."""

        from pymab.plotting import plot_cumulative_regret

        return plot_cumulative_regret(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=self.config.bootstrap,
        )

    def plot_optimal_action_rate(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot replicate-aware optimal-action rates."""

        from pymab.plotting import plot_optimal_action_rate

        return plot_optimal_action_rate(
            self.result,
            output_path=None if output_path is None else Path(output_path),
            show=show,
            band_config=self.config.bootstrap,
        )

    def _summarize(
        self, values: np.ndarray, policy_id: str, metric: str
    ) -> IntervalEstimate:
        config = replace(
            self.config.bootstrap,
            seed=_metric_seed(self.config.bootstrap.seed, policy_id, metric),
        )
        return bootstrap_mean_interval(values, config=config)

    def _summarize_difference(
        self,
        values: np.ndarray,
        policy_index: int,
        baseline_index: int,
        policy_id: str,
        baseline_id: str,
        metric: str,
    ) -> IntervalEstimate:
        return self._summarize(
            values[:, policy_index] - values[:, baseline_index],
            policy_id,
            f"{baseline_id}\x00{metric}",
        )


def compare(
    policies: Mapping[str, Policy | ContextualPolicy],
    *,
    environment: Environment,
    config: ExperimentConfig,
    analysis: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Run one paired experiment and summarize independent replicates."""

    result = Experiment(
        environment=environment,
        policies=policies,
        config=config,
    ).run()
    return BenchmarkResult(
        result=result,
        config=BenchmarkConfig() if analysis is None else analysis,
    )


def _benchmark_metrics(result: SimulationResult) -> dict[str, np.ndarray]:
    return {
        "cumulative_regret": result.cumulative_regret_by_replicate[:, -1, :],
        "total_reward": np.sum(result.rewards, axis=1),
        "optimal_action_rate": np.mean(result.optimal_action_indicator, axis=1),
        "final_simple_regret": result.simple_regret[:, -1, :],
    }


def _add_metric(
    row: SummaryRow,
    name: str,
    estimate: IntervalEstimate,
    *,
    difference: bool = False,
) -> None:
    prefix = f"{name}_difference" if difference else name
    mean_key = f"mean_{prefix}" if not difference else f"mean_{name}_difference"
    row[mean_key] = estimate.estimate
    row[f"{prefix}_standard_error"] = estimate.standard_error
    row[f"{prefix}_ci_lower"] = estimate.ci_lower
    row[f"{prefix}_ci_upper"] = estimate.ci_upper


def _metric_seed(master: int, *parts: str) -> int:
    payload = "\x00".join([str(master), *parts]).encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=b"pymab-ci").digest(),
        byteorder="little",
        signed=False,
    )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "PolicyComparison",
    "PolicySummary",
    "compare",
]
