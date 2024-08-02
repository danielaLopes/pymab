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
    n_bandits: int
    optimistic_initialization: float
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    context_func: Callable

    def __init__(
        self,
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

    @staticmethod
    def get_reward_distribution(name: str) -> Type[RewardDistribution]:
        distributions = {
            "gaussian": GaussianRewardDistribution,
            "bernoulli": BernoulliRewardDistribution,
            "uniform": UniformRewardDistribution,
        }
        if name not in distributions:
            raise ValueError(f"Unknown reward distribution: {name}")
        return distributions[name]

    def _get_actual_reward(self, action_index: int) -> float:
        return self.reward_distribution.get_reward(
            self.Q_values[action_index], self.variance
        )

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        self.current_step += 1
        reward = self._get_actual_reward(chosen_action_index)
        self.total_reward += reward
        self.times_selected[chosen_action_index] += 1
        # Calculate average reward per action without storing all rewards
        self.actions_estimated_reward[chosen_action_index] += (
            reward - self.actions_estimated_reward[chosen_action_index]
        ) / self.times_selected[chosen_action_index]

        return reward

    @property
    def Q_values(self) -> List[float]:
        if self._Q_values is None:
            raise ValueError("Q_values not set yet!")
        return self._Q_values

    @Q_values.setter
    def Q_values(self, Q_values: List[float]) -> None:
        if len(Q_values) != self.n_bandits:
            raise ValueError("Q_values length needs to match n_bandits!")
        self._Q_values = Q_values

    @abstractmethod
    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        pass

    def reset(self):
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
        Plots the distributions of the expected reward for the current step.

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
