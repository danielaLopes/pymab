"""Optional plotting helpers for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pymab.simulation import SimulationResult


def plot_average_reward(
    result: SimulationResult, *, output_path: Path | None = None, show: bool = False
) -> Any:
    """Plot average reward by step with Plotly when installed."""

    return _plot_lines(
        result,
        values=result.average_reward_by_step,
        title="Average reward by step",
        yaxis_title="Average reward",
        output_path=output_path,
        show=show,
    )


def plot_cumulative_regret(
    result: SimulationResult, *, output_path: Path | None = None, show: bool = False
) -> Any:
    """Plot cumulative expected regret by step with Plotly when installed."""

    return _plot_lines(
        result,
        values=result.cumulative_regret,
        title="Cumulative expected regret by step",
        yaxis_title="Cumulative regret",
        output_path=output_path,
        show=show,
    )


def plot_optimal_action_rate(
    result: SimulationResult, *, output_path: Path | None = None, show: bool = False
) -> Any:
    """Plot the optimal-action selection rate by step with Plotly."""

    return _plot_lines(
        result,
        values=result.optimal_action_rate_by_step,
        title="Optimal action rate by step",
        yaxis_title="Optimal action rate",
        output_path=output_path,
        show=show,
    )


def _plot_lines(
    result: SimulationResult,
    *,
    values: np.ndarray,
    title: str,
    yaxis_title: str,
    output_path: Path | None,
    show: bool,
) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Install pymab[plot] to use plotting helpers") from exc

    fig = go.Figure()
    x = list(range(values.shape[0]))
    for index, name in enumerate(result.policy_names):
        fig.add_trace(go.Scatter(x=x, y=values[:, index], mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Step", yaxis_title=yaxis_title)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
    if show:
        fig.show()
    return fig
