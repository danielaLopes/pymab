"""Optional plotting helpers for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pymab._resampling import bootstrap_curve
from pymab.simulation import SimulationResult
from pymab.statistics import BootstrapConfig


def plot_average_reward(
    result: SimulationResult,
    *,
    output_path: Path | None = None,
    show: bool = False,
    band_config: BootstrapConfig | None = None,
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
    band_config: BootstrapConfig | None = None,
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
    band_config: BootstrapConfig | None = None,
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
    band_config: BootstrapConfig | None,
) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Install pymab[plot] to use plotting helpers") from exc

    fig = go.Figure()
    x = list(range(values.shape[0]))
    lower, upper = _bootstrap_band(
        replicate_values,
        config=(
            BootstrapConfig(n_resamples=2_000) if band_config is None else band_config
        ),
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
    replicate_values: object,
    *,
    config: BootstrapConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    settings = BootstrapConfig(n_resamples=2_000) if config is None else config
    if not isinstance(settings, BootstrapConfig):
        raise TypeError("config must be a BootstrapConfig")
    return bootstrap_curve(replicate_values, config=settings)


__all__ = [
    "plot_average_reward",
    "plot_cumulative_regret",
    "plot_optimal_action_rate",
]
