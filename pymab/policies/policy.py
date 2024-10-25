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

    :ivar n_bandits: Number of bandits (actions) available
    :type n_bandits: int
    :ivar optimistic_initialization: Initial Q-value for all actions
    :type optimistic_initialization: float
    :ivar _Q_values: Array storing true Q-values for each action
    :type _Q_values: np.array
    :ivar current_step: Current time step in the learning process
    :type current_step: int
    :ivar total_reward: Cumulative reward obtained so far
    :type total_reward: float
    :ivar times_selected: Number of times each action has been selected
    :type times_selected: np.array
    :ivar actions_estimated_reward: Estimated reward for each action
    :type actions_estimated_reward: np.array
    :ivar variance: Variance of the reward distribution
    :type variance: float
    :ivar reward_distribution: Type of reward distribution
    :type reward_distribution: Type[RewardDistribution]
    :ivar context_func: Function to generate context (if applicable)
    :type context_func: Callable
    :ivar rewards_history: History of rewards for each action
    :type rewards_history: List[List[float]]

    .. note::
        Subclasses should implement the abstract methods to define specific policies.
    """

    @staticmethod
    def get_reward_distribution(name: str) -> Type[RewardDistribution]:
        """
        Get the reward distribution class based on the given name.

        :param name: Name of the reward distribution
        :type name: str
        :return: Reward distribution class
        :rtype: Type[RewardDistribution]
        :raises ValueError: If an unknown reward distribution name is provided

        .. note::
            Supported distributions are 'gaussian', 'bernoulli', and 'uniform'.
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

        :param action_index: Index of the chosen action
        :type action_index: int
        :return: Reward value
        :rtype: float
        """
        return self.reward_distribution.get_reward(
            self.Q_values[action_index], self.variance
        )

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Update the policy based on the chosen action and received reward.

        :param chosen_action_index: Index of the chosen action
        :type chosen_action_index: int
        :return: Reward received for the chosen action
        :rtype: float
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

        :param action_index: Index of the action to update
        :type action_index: int
        :param reward: Reward received for the action
        :type reward: float
        """
        pass

    def _update_sliding_window(self, chosen_action_index: int) -> None:
        """
        Update the sliding window of rewards for a given action.

        :param chosen_action_index: Index of the chosen action
        :type chosen_action_index: int
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

        :return: List of Q-values
        :rtype: List[float]
        :raises ValueError: If Q_values have not been set
        """
        if self._Q_values is None:
            raise ValueError("Q_values not set yet!")
        return self._Q_values

    @Q_values.setter
    def Q_values(self, Q_values: List[float]) -> None:
        """
        Set the true Q-values for all actions.

        :param Q_values: List of Q-values to set
        :type Q_values: List[float]
        :raises ValueError: If the length of Q_values doesn't match n_bandits
        """
        if len(Q_values) != self.n_bandits:
            raise ValueError("Q_values length needs to match n_bandits!")
        self._Q_values = Q_values

    @abstractmethod
    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """
        Select an action based on the policy.

        :return: Tuple containing the chosen action index and the reward
        :rtype: Tuple[int, float]
        """
        pass

    def reset(self):
        """
        Reset the policy to its initial state.

        This method resets all counters, rewards, and estimates to their initial values.
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

        .. note::
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
