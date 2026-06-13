"""Benchmarking helpers for comparing bandit policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from pymab.environments import BanditEnvironment, LinearContextualEnvironment
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult

Environment = BanditEnvironment | LinearContextualEnvironment
BanditPolicy = Policy | ContextualPolicy


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated comparison across repeated simulation seeds."""

    runs: tuple[SimulationResult, ...]
    seeds: tuple[int, ...]
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("at least one run is required")
        if len(self.runs) != len(self.seeds):
            raise ValueError("runs and seeds must have the same length")
        policy_names = self.policy_names
        for run in self.runs:
            if run.policy_names != policy_names:
                raise ValueError("all runs must compare the same policies")

    @property
    def policy_names(self) -> tuple[str, ...]:
        return self.runs[0].policy_names

    @property
    def combined(self) -> SimulationResult:
        """Return a single result with seed repetitions stacked as episodes."""

        q_values = None
        if all(run.q_values is not None for run in self.runs):
            q_values = np.concatenate(
                [np.asarray(run.q_values) for run in self.runs], axis=0
            )
        return SimulationResult(
            rewards=np.concatenate([run.rewards for run in self.runs], axis=0),
            actions=np.concatenate([run.actions for run in self.runs], axis=0),
            expected_rewards=np.concatenate(
                [run.expected_rewards for run in self.runs], axis=0
            ),
            optimal_actions=np.concatenate(
                [run.optimal_actions for run in self.runs], axis=0
            ),
            optimal_values=np.concatenate(
                [run.optimal_values for run in self.runs], axis=0
            ),
            policy_names=self.policy_names,
            q_values=q_values,
        )

    @property
    def best_policy(self) -> str:
        """Policy with the lowest mean final cumulative regret."""

        rows = self.summary()
        best = min(rows, key=lambda row: float(row["mean_cumulative_regret"]))
        return str(best["policy_name"])

    def summary(self) -> list[dict[str, str | float | int]]:
        """Return one summary row per policy with confidence intervals."""

        rows: list[dict[str, str | float | int]] = []
        for policy_index, policy_name in enumerate(self.policy_names):
            cumulative_regret = np.array(
                [float(run.cumulative_regret[-1, policy_index]) for run in self.runs],
                dtype=float,
            )
            total_reward = np.array(
                [
                    float(run.average_reward_by_step[:, policy_index].sum())
                    for run in self.runs
                ],
                dtype=float,
            )
            optimal_action_rate = np.array(
                [
                    float(run.optimal_action_rate_by_step[:, policy_index].mean())
                    for run in self.runs
                ],
                dtype=float,
            )
            regret_margin = confidence_interval_margin(
                cumulative_regret, confidence_level=self.confidence_level
            )
            reward_margin = confidence_interval_margin(
                total_reward, confidence_level=self.confidence_level
            )
            rows.append(
                {
                    "policy_index": policy_index,
                    "policy_name": policy_name,
                    "n_seeds": len(self.seeds),
                    "mean_cumulative_regret": float(np.mean(cumulative_regret)),
                    "cumulative_regret_ci": regret_margin,
                    "mean_total_reward": float(np.mean(total_reward)),
                    "total_reward_ci": reward_margin,
                    "mean_optimal_action_rate": float(np.mean(optimal_action_rate)),
                }
            )
        return rows

    def to_pandas(self) -> Any:
        """Return the benchmark summary as a pandas DataFrame."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pymab[analysis] to use to_pandas().") from exc
        return pd.DataFrame.from_records(self.summary())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable benchmark record."""

        return {
            "seeds": list(self.seeds),
            "confidence_level": self.confidence_level,
            "best_policy": self.best_policy,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }

    def plot_average_reward(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot mean reward by step for the combined benchmark result."""

        from pymab.plotting import plot_average_reward

        return plot_average_reward(
            self.combined,
            output_path=None if output_path is None else Path(output_path),
            show=show,
        )

    def plot_cumulative_regret(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot cumulative regret by step for the combined benchmark result."""

        from pymab.plotting import plot_cumulative_regret

        return plot_cumulative_regret(
            self.combined,
            output_path=None if output_path is None else Path(output_path),
            show=show,
        )

    def plot_optimal_action_rate(
        self, *, output_path: str | Path | None = None, show: bool = False
    ) -> Any:
        """Plot optimal-action rate by step for the combined benchmark result."""

        from pymab.plotting import plot_optimal_action_rate

        return plot_optimal_action_rate(
            self.combined,
            output_path=None if output_path is None else Path(output_path),
            show=show,
        )


def compare(
    policies: Sequence[BanditPolicy],
    *,
    environment: Environment,
    n_steps: int,
    n_episodes: int = 100,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    confidence_level: float = 0.95,
) -> BenchmarkResult:
    """Compare policies across repeated seeds and return benchmark statistics."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    runs = tuple(
        Experiment(
            environment=environment,
            policies=policies,
            config=ExperimentConfig(
                n_episodes=n_episodes,
                n_steps=n_steps,
                seed=int(seed),
            ),
        ).run()
        for seed in seeds
    )
    return BenchmarkResult(
        runs=runs,
        seeds=tuple(int(seed) for seed in seeds),
        confidence_level=confidence_level,
    )


def confidence_interval_margin(
    values: np.ndarray, *, confidence_level: float = 0.95
) -> float:
    """Normal-theory confidence interval margin for repeated benchmark metrics."""

    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if values.size <= 1:
        return 0.0
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(z_score * standard_error)


__all__ = ["BenchmarkResult", "compare", "confidence_interval_margin"]
