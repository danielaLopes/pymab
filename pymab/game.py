from __future__ import annotations

from abc import abstractmethod, ABC
from enum import Enum
from functools import lru_cache
import logging
from pathlib import Path

from plotly.subplots import make_subplots
from tqdm import tqdm
import typing

from joblib import Parallel, delayed
import numpy as np
import plotly.graph_objs as go

from pymab.plot_config import get_line_style, get_default_layout, get_marker_style, get_color_sequence
from pymab.policies.policy import Policy
from pymab.reward_distribution import RewardDistribution
from pymab.static import DEFAULT_ENVIRONMENT_CHANGE_FREQUENCY, DEFAULT_ENVIRONMENT_CHANGE_RATE, \
    DEFAULT_ENVIRONMENT_CHANGE_MAGNITUDE, DEFAULT_ENVIRONMENT_SHIFT_PROBABILITY, DEFAULT_RESULTS_FOLDER

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class EnvironmentChangeType(Enum):
    STATIONARY = "stationary"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    RANDOM_ARM_SWAPPING = "random_arm_swapping"

class EnvironmentChangeMixin(ABC):
    @abstractmethod
    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        pass

class StationaryMixin(EnvironmentChangeMixin):
    """
    Stationary environment's rewards distributions never change, so the Q-values are returned as sampled for each step.
    """
    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        return Q_values

class GradualChangeMixin(EnvironmentChangeMixin):
    """
    Non-stationary environment where the rewards distributions change gradually over time. The change is applied by
    adding a different random value drawn from a normal distribution with 0 mean and {self.change_rate} standard
    deviation to each of the Q-values at each step.
    """
    def __init__(self, change_rate: float):
        self.change_rate = change_rate

    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        return Q_values + np.random.normal(0, self.change_rate, size=Q_values.shape)

class AbruptChangeMixin(EnvironmentChangeMixin):
    """
    Non-stationary environment where the rewards distributions change abruptly and periodically. The change is applied
    by adding a different random value drawn from a normal distribution with 0 mean and {self.change_magnitude} standard
    deviation to each of the Q-values every {self.change_frequency} steps.
    """
    def __init__(self, change_frequency: int, change_magnitude: float):
        self.change_frequency = change_frequency
        self.change_magnitude = change_magnitude

    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        if step % self.change_frequency == 0:
            return Q_values + np.random.normal(0, self.change_magnitude, size=Q_values.shape)
        return Q_values

class RandomArmSwappingMixin(EnvironmentChangeMixin):
    """
    Non-stationary environment where the rewards distributions between arms get swapped abruptly and at random steps.
    """
    def __init__(self, shift_probability: float):
        self.shift_probability = shift_probability

    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        if np.random.random() < self.shift_probability:
            return np.random.permutation(Q_values)
        return Q_values


class Game:
    n_episodes: int
    n_steps: int
    Q_values: np.ndarray
    set_Q_values_flag: bool
    Q_values_mean: float
    Q_values_variance: float
    reward_distribution: Type[RewardDistribution]
    policies: List[Policy]
    n_bandits: int
    rewards: np.ndarray
    actions_selected_by_policy: np.ndarray
    optimal_actions: np.ndarray
    regret_by_policy: np.ndarray
    is_stationary: bool
    results_folder: Path

    def __init__(
        self,
        *,
        n_episodes: int,
        n_steps: int,
        policies: List[Policy],
        n_bandits: int,
        Q_values: List[float] = None,
        Q_values_mean: float = 0.0,
        Q_values_variance: float = 1.0,
        environment_change: Union[EnvironmentChangeType, Type[EnvironmentChangeMixin]] = EnvironmentChangeType.STATIONARY,
        change_params: dict = None,
        results_folder: Path = DEFAULT_RESULTS_FOLDER,
    ) -> None:
        if Q_values is None:
            Q_values = []

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.policies = policies
        self.n_bandits = n_bandits
        self.Q_values = Q_values
        self.set_Q_values_flag = len(self.Q_values) == 0
        self.Q_values_mean = Q_values_mean
        self.Q_values_variance = Q_values_variance
        self.Q_values_history = np.zeros((self.n_episodes * self.n_steps, self.n_bandits))

        reward_distributions = set()
        for policy in self.policies:
            reward_distributions.add(policy.reward_distribution)
        if len(reward_distributions) > 1:
            raise ValueError(
                "All the policies used in a single game should have the same reward distribution"
            )
        self.reward_distribution = reward_distributions.pop()

        self.rewards_by_policy = np.zeros(
            (self.n_episodes, self.n_steps, len(self.policies)), dtype=float
        )
        self.actions_selected_by_policy = np.zeros(
            (self.n_episodes, self.n_steps, len(self.policies))
        )
        self.optimal_actions = np.zeros((self.n_episodes,))
        self.regret_by_policy = np.zeros(
            (self.n_episodes, self.n_steps, len(self.policies))
        )

        self.environment_change = self._create_environment_change(environment_change, change_params)

        self.colors = get_color_sequence()

        self.results_folder = results_folder

    @property
    @lru_cache(maxsize=None)
    def average_rewards_by_step(self) -> np.ndarray:
        return np.mean(self.rewards_by_policy, axis=0)

    @property
    @lru_cache(maxsize=None)
    def average_rewards_by_episode(self) -> np.ndarray:
        return np.mean(self.rewards_by_policy, axis=1)

    def _create_environment_change(self, change_type: Union[EnvironmentChangeType, Type[EnvironmentChangeMixin]],
                                   params: dict = None) -> EnvironmentChangeMixin:
        if isinstance(change_type, type) and issubclass(change_type, EnvironmentChangeMixin):
            return change_type(**params)

        if change_type == EnvironmentChangeType.STATIONARY:
            logger.info('Using `stationary` mode.')
            return StationaryMixin()

        elif change_type == EnvironmentChangeType.GRADUAL:
            logger.info('Using `gradual` mode for environment change.')
            if 'change_rate' not in params:
                logger.warning(f"""Specifying `change_rate` is recommended when using `gradual` mode. Defaulting to 
                {DEFAULT_ENVIRONMENT_CHANGE_FREQUENCY}.""")
            change_rate = params.get('change_rate', DEFAULT_ENVIRONMENT_CHANGE_RATE)
            return GradualChangeMixin(change_rate)

        elif change_type == EnvironmentChangeType.ABRUPT:
            logger.info('Using `abrupt` mode for environment change.')
            if 'change_frequency' not in params:
                logger.warning(f"""Specifying `change_frequency` is recommended when using `abrupt` mode. Defaulting to 
                {DEFAULT_ENVIRONMENT_CHANGE_FREQUENCY}.""")
            if 'change_magnitude' not in params:
                logger.warning(
                    f"""Specifying `change_magnitude` is recommended when using `abrupt` mode. Defaulting to 
                    {DEFAULT_ENVIRONMENT_CHANGE_MAGNITUDE}.""")
            change_frequency = params.get('change_frequency', DEFAULT_ENVIRONMENT_CHANGE_FREQUENCY)
            change_magnitude = params.get('change_magnitude', DEFAULT_ENVIRONMENT_CHANGE_MAGNITUDE)
            return AbruptChangeMixin(change_frequency, change_magnitude)

        elif change_type == EnvironmentChangeType.RANDOM_ARM_SWAPPING:
            logger.info('Using `random arm swapping` mode for environment change.')
            if 'shift_probability' not in params:
                logger.warning(f"""Specifying `shift_probability` is recommended when using `random arm swapping` mode. Defaulting to 
                {DEFAULT_ENVIRONMENT_SHIFT_PROBABILITY}.""")
            shift_probability = params.get('shift_probability', DEFAULT_ENVIRONMENT_SHIFT_PROBABILITY)
            return RandomArmSwappingMixin(shift_probability)

        else:
            raise ValueError(f"Unknown environment change type: {change_type}")

    @property
    @lru_cache(maxsize=None)
    def cumulative_regret_by_step(self) -> np.ndarray:
        """
        Calculate the cumulative regret for each policy. The regret measures how much worse a chosen strategy performs
        compared to the optimal strategy. It quantifies the difference between the reward obtained by the policy and
        the reward that would have been obtained by always selecting the best possible action

        Returns:
            np.ndarray: The cumulative regret for each policy.
        """
        return np.cumsum(np.mean(self.regret_by_policy, axis=0), axis=0)

    def game_loop(self) -> None:
        logger.info(f"Starting game loop for {self.n_episodes} episodes, {self.n_steps} in each episode, and analysing {len(self.policies)} policies ...")
        for episode in range(self.n_episodes):
            self.new_episode(episode)
            self.optimal_actions[episode] = np.argmax(self.Q_values)
            optimal_reward = self.Q_values[int(self.optimal_actions[episode])]
            # for step in tqdm(range(self.n_steps), desc="Running steps...", total=self.n_steps):
            for step in range(self.n_steps):
                # print("\n\n========= Episode: ", episode, "Step: ", step)
                current_step = episode * self.n_steps + step
                self.Q_values_history[current_step] = self.Q_values

                self._update_environment(episode * self.n_steps + step)
                # for policy_index, policy in tqdm(enumerate(self.policies), desc="Running game for each policy...", total=len(self.policies)):
                for policy_index, policy in enumerate(self.policies):
                    context = policy.context_func()
                    action, reward = policy.select_action(context=context)
                    self.rewards_by_policy[episode, step, policy_index] = reward
                    self.actions_selected_by_policy[episode, step, policy_index] = (
                        action
                    )
                    self.regret_by_policy[episode, step, policy_index] = (
                        optimal_reward - reward
                    )

    def _generate_initial_Q_values(self) -> None:
        self.Q_values = self.reward_distribution.generate_Q_values(
            self.Q_values_mean, self.Q_values_variance, self.n_bandits
        )

    def _moving_average(self, data: np.ndarray, smooth_factor: int):
        return np.convolve(data, np.ones(smooth_factor) / smooth_factor, mode="valid")

    def new_episode(self, episode_idx: int) -> None:
        if episode_idx == 0 and self.set_Q_values_flag:
            self._generate_initial_Q_values()

        for policy in self.policies:
            policy.Q_values = self.Q_values
            policy.reset()

    def plot_average_reward_by_step(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for policy_index, policy in enumerate(self.policies):
            fig.add_trace(go.Scatter(
                x=list(range(self.n_steps)),
                y=self.total_rewards_by_step[:, policy_index],
                mode='lines',
                name=repr(policy),
                line=get_line_style(self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Cumulative reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Steps",
                yaxis_title="Cumulative reward"
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"average_reward_by_step_{plot_name}.html")

    def plot_average_reward_by_episode(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for policy_index, policy in enumerate(self.policies):
            fig.add_trace(go.Scatter(
                x=list(range(self.n_episodes)),
                y=self.average_rewards_by_episode[:, policy_index],
                mode='lines',
                name=repr(policy),
                line=dict(color=self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Average reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Episodes",
                yaxis_title="Average reward",
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"average_reward_by_episode_{plot_name}.html")

    def plot_average_reward_by_step_smoothed(self, smooth_factor: int = 50, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for policy_index, policy in enumerate(self.policies):
            smoothed_data = self._moving_average(
                self.average_rewards_by_step[:, policy_index], smooth_factor
            )
            fig.add_trace(go.Scatter(
                x=list(range(len(smoothed_data))),
                y=smoothed_data,
                mode='lines',
                name=repr(policy),
                line=dict(color=self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Smoothed average reward (factor: {smooth_factor}) during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Steps",
                yaxis_title="Average reward",
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"average_reward_by_step_smoothed_{plot_name}.html")

    def plot_bandit_selection_evolution(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        colorscale = get_color_sequence()[:self.n_bandits]

        for policy_index, policy in enumerate(self.policies):
            arm_selections = self.actions_selected_by_policy[:, :, policy_index].flatten()

            for arm in range(self.n_bandits):
                arm_mask = arm_selections == arm
                fig.add_trace(go.Scatter(
                    x=np.arange(self.n_episodes * self.n_steps)[arm_mask],
                    y=np.full(np.sum(arm_mask), policy_index),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colorscale[arm],
                    ),
                    name=f'{repr(policy)} - Arm {arm}',
                    showlegend=policy_index == 0
                ))

        fig.update_layout(
            **get_default_layout(
                title="Arm Selections by Policy Over Time",
                xaxis_title="Steps",
                yaxis_title="Policy",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(self.policies))),
                    ticktext=[repr(policy) for policy in self.policies]
                ),
                height=100 * len(self.policies) + 200
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"bandit_selection_evolution_{plot_name}.html")

    def plot_cumulative_regret_by_step(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for policy_index, policy in enumerate(self.policies):
            fig.add_trace(go.Scatter(
                x=list(range(self.n_steps)),
                y=self.cumulative_regret_by_step[:, policy_index],
                mode='lines',
                name=repr(policy),
                line=dict(color=self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Cumulative regret during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Steps",
                yaxis_title="Cumulative Regret",
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"cumulative_regret_by_step_{plot_name}.html")

    def plot_Q_values(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(self.Q_values))),
            y=self.Q_values,
            mode='markers',
            marker=get_marker_style(self.colors[0])
        ))

        fig.update_layout(
            **get_default_layout(
                title="Q values for each action",
                xaxis_title="Actions",
                yaxis_title="Q value"
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"Q_values_{plot_name}.html")

    def plot_Q_values_evolution_by_bandit_first_episode(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for bandit_index in range(self.n_bandits):
            bandit_Q_values = self.Q_values_history[:self.n_steps, bandit_index]

            fig.add_trace(go.Scatter(
                x=list(range(self.n_steps)),
                y=bandit_Q_values,
                mode='lines',
                name=f'Bandit {bandit_index}',
                line=dict(color=self.colors[bandit_index])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Q-values evolution for Bandits during the first episode ({self.n_steps} steps)",
                xaxis_title="Steps",
                yaxis_title="Q-value"
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"Q_values_evolution_by_bandwidth_first_episode_{plot_name}.html")

    def plot_total_reward_by_step(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        for policy_index, policy in enumerate(self.policies):
            fig.add_trace(go.Scatter(
                x=list(range(self.n_steps)),
                y=self.total_rewards_by_step[:, policy_index],
                mode='lines',
                name=repr(policy),
                line=dict(color=self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Cumulative reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Steps",
                yaxis_title="Cumulative reward",
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"total_reward_by_step_{plot_name}.html")

    def plot_rate_optimal_actions_by_step(self, save: bool = True, plot_name: str = "") -> None:
        fig = go.Figure()

        optimal_actions_expanded = np.repeat(
            self.optimal_actions[:, np.newaxis], self.n_steps, axis=1
        )
        optimal_action_selections = (
            self.actions_selected_by_policy
            == optimal_actions_expanded[:, :, np.newaxis]
        )
        percentage_optimal_by_step = np.mean(optimal_action_selections, axis=0) * 100

        for policy_index, policy in enumerate(self.policies):
            fig.add_trace(go.Scatter(
                x=list(range(self.n_steps)),
                y=percentage_optimal_by_step[:, policy_index],
                mode='lines',
                name=repr(policy),
                line=dict(color=self.colors[policy_index % len(self.colors)])
            ))

        fig.update_layout(
            **get_default_layout(
                title=f"Percentage of optimal actions chosen during the {self.n_steps} steps for {self.n_episodes} episodes",
                xaxis_title="Steps",
                yaxis_title="% Optimal actions",
            )
        )

        fig.show()
        if save:
            fig.write_html(self.results_folder / f"rate_optimal_actions_by_step_{plot_name}.html")

    @property
    @lru_cache(maxsize=None)
    def total_rewards_by_step(self) -> np.ndarray:
        return np.cumsum(np.mean(self.rewards_by_policy, axis=0), axis=0)

    def _update_non_stationary_Q_values(self, episode_idx: int) -> None:
        if self.change_frequency and episode_idx % self.change_frequency == 0:
            self.Q_values += np.random.normal(0, self.change_magnitude, self.n_bandits)
            print("Q_values updated", self.Q_values)

    def _update_environment(self, step: int):
        self.Q_values = self.environment_change.apply_change(self.Q_values, step)