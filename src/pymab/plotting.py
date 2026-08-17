"""Optional plotting helpers for simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from pymab.simulation import SimulationResult
from pymab.validation import finite_float, positive_integer


@dataclass(frozen=True)
class BootstrapBandConfig:
    """Configuration for replicate-level plot uncertainty bands."""

    confidence_level: float = 0.95
    n_resamples: int = 2_000
    seed: int = 0
    max_output_elements: int = 1_000_000

    def __post_init__(self) -> None:
        confidence = finite_float(self.confidence_level, name="confidence_level")
        if not 0 < confidence < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self, "n_resamples", positive_integer(self.n_resamples, name="n_resamples")
        )
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "max_output_elements",
            positive_integer(self.max_output_elements, name="max_output_elements"),
        )
        if self.max_output_elements < self.n_resamples:
            raise ValueError(
                "max_output_elements must be at least as large as n_resamples"
            )


def plot_average_reward(
    result: SimulationResult,
    *,
    output_path: Path | None = None,
    show: bool = False,
    band_config: BootstrapBandConfig | None = None,
) -> Any:
    """Plot average reward by step with Plotly when installed."""

    return _plot_lines(
        result,
        values=result.average_reward_by_step,
        replicate_values=result.rewards,
        title="Average reward by step",
        yaxis_title="Average reward",
        output_path=output_path,
        show=show,
        band_config=band_config,
    )


def plot_cumulative_regret(
    result: SimulationResult,
    *,
    output_path: Path | None = None,
    show: bool = False,
    band_config: BootstrapBandConfig | None = None,
) -> Any:
    """Plot cumulative expected regret by step with Plotly when installed."""

    return _plot_lines(
        result,
        values=result.cumulative_regret,
        replicate_values=result.cumulative_regret_by_replicate,
        title="Cumulative expected regret by step",
        yaxis_title="Cumulative regret",
        output_path=output_path,
        show=show,
        band_config=band_config,
    )


def plot_optimal_action_rate(
    result: SimulationResult,
    *,
    output_path: Path | None = None,
    show: bool = False,
    band_config: BootstrapBandConfig | None = None,
) -> Any:
    """Plot the optimal-action selection rate by step with Plotly."""

    return _plot_lines(
        result,
        values=result.optimal_action_rate_by_step,
        replicate_values=np.asarray(result.optimal_action_indicator, dtype=float),
        title="Optimal action rate by step",
        yaxis_title="Optimal action rate",
        output_path=output_path,
        show=show,
        band_config=band_config,
    )


def _plot_lines(
    result: SimulationResult,
    *,
    values: np.ndarray,
    replicate_values: np.ndarray,
    title: str,
    yaxis_title: str,
    output_path: Path | None,
    show: bool,
    band_config: BootstrapBandConfig | None,
) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Install pymab[plot] to use plotting helpers") from exc

    fig = go.Figure()
    x = list(range(values.shape[0]))
    lower, upper = _bootstrap_band(
        replicate_values,
        config=BootstrapBandConfig() if band_config is None else band_config,
    )
    for index, name in enumerate(result.policy_ids):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper[:, index],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower[:, index],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(100, 100, 100, 0.12)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=x, y=values[:, index], mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Step", yaxis_title=yaxis_title)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
    if show:
        fig.show()
    return fig


def _bootstrap_band(
    replicate_values: np.ndarray,
    *,
    config: BootstrapBandConfig | None = None,
    confidence_level: float | None = None,
    n_resamples: int | None = None,
    seed: int | None = None,
    max_output_elements: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if config is not None and any(
        value is not None
        for value in (confidence_level, n_resamples, seed, max_output_elements)
    ):
        raise ValueError("config cannot be combined with individual band settings")
    settings = config or BootstrapBandConfig(
        confidence_level=0.95 if confidence_level is None else confidence_level,
        n_resamples=2_000 if n_resamples is None else n_resamples,
        seed=0 if seed is None else seed,
        max_output_elements=(
            1_000_000 if max_output_elements is None else max_output_elements
        ),
    )
    if replicate_values.ndim != 3:
        raise ValueError("replicate_values must have shape (replicate, step, policy)")
    if replicate_values.size == 0 or not np.all(np.isfinite(replicate_values)):
        raise ValueError("replicate_values must be non-empty and finite")
    if replicate_values.shape[0] < 2:
        mean = np.mean(replicate_values, axis=0)
        return mean, mean
    n_steps = replicate_values.shape[1]
    n_policies = replicate_values.shape[2]
    policies_per_chunk = max(
        1, min(n_policies, settings.max_output_elements // settings.n_resamples)
    )
    steps_per_chunk = max(
        1,
        settings.max_output_elements // (settings.n_resamples * policies_per_chunk),
    )
    lower = np.empty((n_steps, n_policies), dtype=float)
    upper = np.empty_like(lower)
    alpha = (1.0 - settings.confidence_level) / 2.0
    for policy_start in range(0, n_policies, policies_per_chunk):
        policy_stop = min(policy_start + policies_per_chunk, n_policies)
        for start in range(0, n_steps, steps_per_chunk):
            stop = min(start + steps_per_chunk, n_steps)
            rng = np.random.default_rng(settings.seed)
            means = np.empty(
                (settings.n_resamples, stop - start, policy_stop - policy_start)
            )
            for resample in range(settings.n_resamples):
                indices = rng.integers(
                    0,
                    replicate_values.shape[0],
                    size=replicate_values.shape[0],
                )
                means[resample] = np.mean(
                    replicate_values[indices, start:stop, policy_start:policy_stop],
                    axis=0,
                )
            quantiles = np.quantile(means, [alpha, 1.0 - alpha], axis=0)
            lower[start:stop, policy_start:policy_stop] = quantiles[0]
            upper[start:stop, policy_start:policy_stop] = quantiles[1]
    return lower, upper


__all__ = [
    "BootstrapBandConfig",
    "plot_average_reward",
    "plot_cumulative_regret",
    "plot_optimal_action_rate",
]
