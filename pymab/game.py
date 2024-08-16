from __future__ import annotations

from abc import abstractmethod, ABC
from enum import Enum
from functools import lru_cache
import logging

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
    DEFAULT_ENVIRONMENT_CHANGE_MAGNITUDE

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class EnvironmentChangeType(Enum):
    STATIONARY = "stationary"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"

class EnvironmentChangeMixin(ABC):
    @abstractmethod
    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        pass

class StationaryMixin(EnvironmentChangeMixin):
    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        return Q_values

class GradualChangeMixin(EnvironmentChangeMixin):
    def __init__(self, change_rate: float):
        self.change_rate = change_rate

    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        return Q_values + np.random.normal(0, self.change_rate, size=Q_values.shape)

class AbruptChangeMixin(EnvironmentChangeMixin):
    def __init__(self, change_frequency: int, change_magnitude: float):
        self.change_frequency = change_frequency
        self.change_magnitude = change_magnitude

    def apply_change(self, Q_values: np.ndarray, step: int) -> np.ndarray:
        if step % self.change_frequency == 0:
            return Q_values + np.random.normal(0, self.change_magnitude, size=Q_values.shape)
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
        change_params: dict = None
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

    def plot_average_reward_by_step(self) -> None:
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

    def plot_average_reward_by_episode(self) -> None:
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

    def plot_average_reward_by_step_smoothed(self, smooth_factor: int = 50) -> None:
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

    def plot_cumulative_regret_by_step(self) -> None:
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

    def plot_optimal_arm_evolution(self):
        optimal_arms = np.argmax(self.Q_values_history, axis=1)

        scatter = go.Scatter(
            x=list(range(self.n_episodes * self.n_steps)),
            y=optimal_arms,
            mode='markers',
            marker=dict(
                size=5,
                color=optimal_arms,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Arm Number')
            ),
            name='Optimal Arm'
        )

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02,
                            row_heights=[0.7, 0.3])

        fig.add_trace(scatter, row=1, col=1)

        heatmap = go.Heatmap(
            z=self.Q_values_history.T,
            x=list(range(self.n_episodes * self.n_steps)),
            y=list(range(self.n_bandits)),
            colorscale='Viridis',
            name='Q-values'
        )
        fig.add_trace(heatmap, row=2, col=1)

        fig.update_layout(
            title='Evolution of Optimal Arm and Q-values Over Time',
            xaxis_title='Time Steps',
            yaxis_title='Arm Number',
            height=800,
            width=1200,
            yaxis=dict(range=[-0.5, self.n_bandits - 0.5])
        )

        fig.update_yaxes(title_text="Arm Number", row=2, col=1)

        fig.show()

    def plot_Q_values(self) -> None:
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

    def plot_total_reward_by_step(self) -> None:
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

    def plot_rate_optimal_actions_by_step(self) -> None:
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