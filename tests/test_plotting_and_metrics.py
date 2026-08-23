from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from pymab import BanditEnvironment, Experiment, ExperimentConfig, compare
from pymab.benchmarking import BenchmarkConfig
from pymab.metrics import (
    average_reward_by_step,
    best_arm_identification_rate_by_step,
    cumulative_regret_by_step,
    cumulative_reward_by_step,
    expected_regret_by_step,
    optimal_action_rate_by_step,
    simple_regret_by_step,
)
from pymab.plotting import (
    _bootstrap_band,
    _plot_lines,
    plot_average_reward,
    plot_cumulative_regret,
    plot_optimal_action_rate,
)
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.statistics import BootstrapConfig


def _result():
    return Experiment(
        environment=BanditEnvironment(means=np.array([0.0, 1.0])),
        policies={"random": RandomPolicy(n_arms=2), "ucb": UCBPolicy(n_arms=2)},
        config=ExperimentConfig(horizon=4, n_replicates=3, seed=1),
    ).run()


def test_metric_functions_delegate_to_validated_result() -> None:
    result = _result()
    np.testing.assert_array_equal(
        average_reward_by_step(result), result.average_reward_by_step
    )
    np.testing.assert_array_equal(
        cumulative_reward_by_step(result), result.cumulative_reward_by_step
    )
    np.testing.assert_array_equal(expected_regret_by_step(result), result.regret)
    np.testing.assert_array_equal(
        cumulative_regret_by_step(result), result.cumulative_regret
    )
    np.testing.assert_array_equal(
        optimal_action_rate_by_step(result), result.optimal_action_rate_by_step
    )
    np.testing.assert_array_equal(
        simple_regret_by_step(result), np.mean(result.simple_regret, axis=0)
    )
    np.testing.assert_array_equal(
        best_arm_identification_rate_by_step(result),
        np.mean(result.recommendation_is_optimal, axis=0),
    )


def test_bootstrap_band_validation_and_single_replicate() -> None:
    values = np.ones((1, 3, 2))
    lower, upper = _bootstrap_band(
        values, config=BootstrapConfig(n_resamples=5, max_chunk_elements=5)
    )
    np.testing.assert_array_equal(lower, np.ones((3, 2)))
    np.testing.assert_array_equal(upper, np.ones((3, 2)))
    with pytest.raises(ValueError, match="3D"):
        _bootstrap_band(np.ones((2, 2)))


def test_bootstrap_band_chunking_preserves_results() -> None:
    values = np.random.default_rng(1).normal(size=(4, 7, 5))
    unchunked = _bootstrap_band(
        values,
        config=BootstrapConfig(
            n_resamples=40,
            seed=3,
            max_chunk_elements=10_000,
        ),
    )
    chunked = _bootstrap_band(
        values,
        config=BootstrapConfig(
            n_resamples=40,
            seed=3,
            max_chunk_elements=80,
        ),
    )
    np.testing.assert_array_equal(chunked[0], unchunked[0])
    np.testing.assert_array_equal(chunked[1], unchunked[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 1},
        {"n_resamples": 0},
        {"seed": True},
        {"max_chunk_elements": 0},
        {"n_resamples": 100, "max_chunk_elements": 50},
    ],
)
def test_bootstrap_band_config_validation(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        BootstrapConfig(**kwargs)


def test_bootstrap_band_rejects_invalid_configuration_type() -> None:
    with pytest.raises(TypeError, match="BootstrapConfig"):
        _bootstrap_band(np.ones((2, 2, 1)), config=object())  # type: ignore[arg-type]


def test_plotting_extra_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def missing_plotly(name, *args, **kwargs):
        if name == "plotly.graph_objects":
            raise ImportError("unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_plotly)
    result = _result()
    with pytest.raises(ImportError, match=r"pymab\[plot\]"):
        _plot_lines(
            result,
            values=result.average_reward_by_step,
            replicate_values=result.rewards,
            title="test",
            yaxis_title="value",
            output_path=None,
            show=False,
            band_config=BootstrapConfig(n_resamples=5),
        )


@pytest.mark.optional
@pytest.mark.parametrize(
    "plotter",
    [plot_average_reward, plot_cumulative_regret, plot_optimal_action_rate],
)
def test_plotters_create_html(plotter, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    destination = tmp_path / f"{plotter.__name__}.html"
    figure = plotter(_result(), output_path=destination, show=False)
    assert destination.exists()
    assert len(figure.data) == 6


@pytest.mark.optional
def test_benchmark_plot_wrappers(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    benchmark = compare(
        {"random": RandomPolicy(n_arms=2), "ucb": UCBPolicy(n_arms=2)},
        environment=BanditEnvironment(means=np.array([0.0, 1.0])),
        config=ExperimentConfig(horizon=3, n_replicates=2, seed=1),
        analysis=BenchmarkConfig(bootstrap=BootstrapConfig(n_resamples=5)),
    )
    assert (
        benchmark.plot_average_reward(output_path=tmp_path / "reward.html") is not None
    )
    assert (
        benchmark.plot_cumulative_regret(output_path=tmp_path / "regret.html")
        is not None
    )
    assert (
        benchmark.plot_optimal_action_rate(output_path=tmp_path / "optimal.html")
        is not None
    )


@pytest.mark.optional
def test_plotter_show_delegates_to_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    shown = False

    def fake_show(self) -> None:
        nonlocal shown
        shown = True

    monkeypatch.setattr(go.Figure, "show", fake_show)
    plot_average_reward(
        _result(),
        show=True,
        band_config=BootstrapConfig(n_resamples=5),
    )
    assert shown
