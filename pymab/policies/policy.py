from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import typing

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import beta, norm

from pymab.reward_distribution import (
    RewardDistribution,
    GaussianRewardDistribution,
    UniformRewardDistribution,
    BernoulliRewardDistribution,
)

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


def no_context_func():
    """
    Dummy context function that does not generate any context.
    """
    pass


class Policy(ABC):
    """
    Abstract base class for multi-armed bandit policies.

    This class provides a framework for implementing various bandit algorithms.
    It handles common functionality such as reward tracking, action selection,
    and policy updates.

    Args:
        n_bandits: Number of bandits (actions) available.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution ("gaussian", "bernoulli", or "uniform"). 
            Defaults to "gaussian".
        context_func: Function to generate context. Defaults to no_context_func.

    Attributes:
        n_bandits (int): Number of available actions.
        optimistic_initialization (float): Initial value for all action estimates.
        _Q_values (np.ndarray): True Q-values for each action.
        current_step (int): Current time step in the learning process.
        total_reward (float): Cumulative reward obtained.
        times_selected (np.ndarray): Selection count for each action.
        actions_estimated_reward (np.ndarray): Current reward estimates.
        variance (float): Reward distribution variance.
        reward_distribution (Type[RewardDistribution]): Reward distribution class.
        context_func (Callable): Context generation function.
        rewards_history (List[List[float]]): History of rewards per action.
    """
    
    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        context_func: Callable = no_context_func,
    ) -> None:
        self.n_bandits = n_bandits
        self.optimistic_initialization = optimistic_initialization
        self._Q_values = None
        self.current_step = 0
        self.total_reward = 0
        self.variance = variance
        self.reward_distribution = self.get_reward_distribution(reward_distribution)
        self.times_selected = np.zeros(self.n_bandits)
        self.actions_estimated_reward = np.full(
            self.n_bandits, self.optimistic_initialization, dtype=float
        )
        self.context_func = context_func
        self.rewards_history = [[] for _ in range(n_bandits)]

    @staticmethod
    def get_reward_distribution(name: str) -> Type[RewardDistribution]:
        """
        Get the reward distribution class based on the given name.

        Args:
            name: Name of the reward distribution.

        Returns:
            The corresponding reward distribution class.

        Raises:
            ValueError: If an unknown distribution name is provided.

        Note:
            Supported distributions are:
            - 'gaussian': Normal distribution
            - 'bernoulli': Binary distribution
            - 'uniform': Uniform distribution
        """
        distributions = {
            "gaussian": GaussianRewardDistribution,
            "bernoulli": BernoulliRewardDistribution,
            "uniform": UniformRewardDistribution,
        }
        if name not in distributions:
            raise ValueError(f"Unknown reward distribution: {name}")
        return distributions[name]

    def _get_actual_reward(self, action_index: int) -> float:
        """
        Get the actual reward for a given action.

        Args:
            action_index: Index of the chosen action.

        Returns:
            The reward value sampled from the distribution.
        """
        return self.reward_distribution.get_reward(
            self.Q_values[action_index], self.variance
        )

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Update the policy based on the chosen action and received reward.

        Args:
            chosen_action_index: Index of the chosen action.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The reward received for the chosen action.
        """
        self.current_step += 1
        reward = self._get_actual_reward(chosen_action_index)
        self.total_reward += reward
        self.times_selected[chosen_action_index] += 1

        self.rewards_history[chosen_action_index].append(reward)

        self._update_estimate(chosen_action_index, reward)

        return reward

    @abstractmethod
    def _update_estimate(self, action_index: int, reward: float) -> None:
        """
        Update the estimated reward for a given action.

        Args:
            action_index: Index of the action to update.
            reward: Reward received for the action.
        """
        pass

    def _update_sliding_window(self, chosen_action_index: int) -> None:
        """
        Update the sliding window of rewards for a given action.

        Args:
            chosen_action_index: Index of the chosen action.

        Note:
            This method maintains a fixed-size window of recent rewards
            and updates the estimated reward based on this window.
        """
        if len(self.rewards_history[chosen_action_index]) > self.sliding_window_size:
            self.rewards_history[chosen_action_index] = self.rewards_history[
                chosen_action_index
            ][-self.sliding_window_size :]

        self.actions_estimated_reward[chosen_action_index] = np.mean(
            self.rewards_history[chosen_action_index]
        )

    @property
    def Q_values(self) -> List[float]:
        """
        Get the true Q-values for all actions.

        Returns:
            List of true Q-values.

        Raises:
            ValueError: If Q-values haven't been set.
        """
        if self._Q_values is None:
            raise ValueError("Q_values not set yet!")
        return self._Q_values

    @Q_values.setter
    def Q_values(self, Q_values: List[float]) -> None:
        """
        Set the true Q-values for all actions.

        Args:
            Q_values: List of Q-values to set.

        Raises:
            ValueError: If length of Q_values doesn't match n_bandits.
        """
        if len(Q_values) != self.n_bandits:
            raise ValueError("Q_values length needs to match n_bandits!")
        self._Q_values = Q_values

    @abstractmethod
    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """
        Select an action based on the policy.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A tuple containing:
            - Index of the chosen action (int)
            - Reward received for the action (float)
        """
        pass

    def reset(self):
        """
        Reset the policy to its initial state.

        This method resets:
        - Current step counter
        - Total reward
        - Action selection counts
        - Estimated rewards
        - Reward history
        """
        self.current_step = 0
        self.total_reward = 0
        self.times_selected = np.zeros(self.n_bandits)
        self.actions_estimated_reward = np.full(
            self.n_bandits, self.optimistic_initialization, dtype=float
        )

    def __repr__(self) -> str:
        return self.__class__.__name__

    def plot_distribution(self) -> None:
        """
        Plot the distributions of the expected reward for the current step.

        This method visualizes the reward distributions for all actions, showing
        both the estimated rewards and true rewards (if known).

        Note:
            This method handles both Bernoulli and Gaussian reward distributions.
        """
        fig, axes = plt.subplots(
            1, self.n_bandits, figsize=(15, 6), constrained_layout=True
        )
        x_range = (
            np.linspace(0, 1, 1000)
            if isinstance(self.reward_distribution, BernoulliRewardDistribution)
            else np.linspace(-2, 2, 1000)
        )

        for i in range(self.n_bandits):
            if issubclass(self.reward_distribution, BernoulliRewardDistribution):
                a, b = (
                    self.times_selected[i] + 1,
                    self.times_selected[i] - self.actions_estimated_reward[i] + 1,
                )
                y = beta.pdf(x_range, a, b)
            elif issubclass(self.reward_distribution, GaussianRewardDistribution):
                mean, std_dev = self.actions_estimated_reward[i], np.sqrt(self.variance)
                y = norm.pdf(x_range, mean, std_dev)
            else:
                raise ValueError("Unsupported reward distribution")

            axes[i].plot(x_range, y, label=f"Arm {i} Posterior")
            axes[i].axvline(
                self.Q_values[i], color="r", linestyle="--", label="True Reward"
            )
            axes[i].legend()
            axes[i].set_title(f"Arm {i} - Step {self.current_step}")
            axes[i].text(
                0.5,
                -0.1,
                f"Mean espected reward: {round(self.actions_estimated_reward[i], 2)}, Std: {round(np.sqrt(self.variance), 2)}",
                transform=axes[i].transAxes,
                ha="center",
                va="top",
            )

        fig.suptitle(f"Reward distribution for {self.__class__.__name__}")
        plt.show()
