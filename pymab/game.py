"""Compatibility wrapper around the v1 experiment API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from warnings import warn

import numpy as np

from pymab.environments import (
    BanditEnvironment,
    EnvironmentChangeType,
    make_dynamics,
)
from pymab.plotting import plot_average_reward, plot_cumulative_regret
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult
from pymab.static import DEFAULT_RESULTS_FOLDER


class Game:
    """Backward-compatible facade for older PyMAB examples.

    New code should use :class:`pymab.simulation.Experiment` directly.
    """

    def __init__(
        self,
        *,
        n_episodes: int,
        n_steps: int,
        policies: list[Any],
        n_bandits: int,
        Q_values: list[float] | np.ndarray | None = None,
        Q_values_mean: float = 0.0,
        Q_values_variance: float = 1.0,
        environment_change: EnvironmentChangeType
        | str = EnvironmentChangeType.STATIONARY,
        change_params: dict[str, Any] | None = None,
        results_folder: Path = DEFAULT_RESULTS_FOLDER,
        seed: int | None = None,
        reward_distribution: str = "gaussian",
    ) -> None:
        warn(
            "Game is deprecated; use BanditEnvironment, ExperimentConfig, and "
            "Experiment instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.policies = policies
        self.n_bandits = n_bandits
        self.results_folder = results_folder
        self.seed = seed
        dynamics = make_dynamics(environment_change, change_params)
        if Q_values is None:
            rng = np.random.default_rng(seed)
            self.environment = BanditEnvironment.from_distribution(
                n_arms=n_bandits,
                reward_distribution=reward_distribution,
                q_mean=Q_values_mean,
                q_scale=Q_values_variance,
                dynamics=dynamics,
                rng=rng,
            )
        else:
            self.environment = BanditEnvironment(
                q_values=np.asarray(Q_values, dtype=float),
                dynamics=dynamics,
            )
        self.result: SimulationResult | None = None

    def game_loop(self) -> None:
        """Run the experiment and store the result."""

        config = ExperimentConfig(
            n_episodes=self.n_episodes, n_steps=self.n_steps, seed=self.seed
        )
        self.result = Experiment(
            environment=self.environment,
            policies=self.policies,
            config=config,
        ).run()

    @property
    def rewards_by_policy(self) -> np.ndarray:
        return self._result.rewards

    @property
    def actions_selected_by_policy(self) -> np.ndarray:
        return self._result.actions

    @property
    def regret_by_policy(self) -> np.ndarray:
        return self._result.regret

    @property
    def Q_values_history(self) -> np.ndarray:
        if self._result.q_values is None:
            raise ValueError("q_values are not available for this result")
        return self._result.q_values.reshape(self.n_episodes * self.n_steps, -1)

    @property
    def Q_values(self) -> np.ndarray:
        return self.environment.q_values

    @property
    def optimal_actions(self) -> np.ndarray:
        return self._result.optimal_actions[:, 0]

    @property
    def average_rewards_by_step(self) -> np.ndarray:
        return self._result.average_reward_by_step

    @property
    def average_rewards_by_episode(self) -> np.ndarray:
        return cast(np.ndarray, np.mean(self._result.rewards, axis=1))

    @property
    def cumulative_regret_by_step(self) -> np.ndarray:
        return self._result.cumulative_regret

    @property
    def total_rewards_by_step(self) -> np.ndarray:
        return self._result.cumulative_reward_by_step

    def plot_average_reward_by_step(
        self,
        save: bool = True,
        plot_name: str = "",
        plot_config: dict[str, Any] | None = None,
    ) -> None:
        output = (
            self.results_folder / f"average_reward_by_step_{plot_name}.html"
            if save
            else None
        )
        plot_average_reward(self._result, output_path=output, show=True)

    def plot_cumulative_regret_by_step(
        self,
        save: bool = True,
        plot_name: str = "",
        plot_config: dict[str, Any] | None = None,
    ) -> None:
        output = (
            self.results_folder / f"cumulative_regret_by_step_{plot_name}.html"
            if save
            else None
        )
        plot_cumulative_regret(self._result, output_path=output, show=True)

    @property
    def _result(self) -> SimulationResult:
        if self.result is None:
            raise ValueError("game_loop must be called before reading results")
        return self.result


__all__ = ["EnvironmentChangeType", "Game"]
