from __future__ import annotations

import logging

import numpy as np
import typing

from pymab.policies.mixins.stationarity_mixins import StationaryPolicyMixin
from pymab.policies.policy import Policy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class GreedyPolicy(StationaryPolicyMixin, Policy):
    """
    Implements a Greedy policy for multi-armed bandit problems.

    The Greedy policy always selects the action with the highest estimated reward.
    It inherits from StationaryPolicyMixin and Policy, combining stationary behavior
    with basic policy functionality.

    Attributes:
        n_bandits (int): Number of bandit arms.
        optimistic_initialization (float): Initial optimistic value for estimated rewards.
        Q_values (np.array): True Q-values for each arm (set externally).
        current_step (int): Current time step in the learning process.
        total_reward (float): Cumulative reward obtained so far.
        times_selected (np.array): Number of times each arm has been selected.
        actions_estimated_reward (np.array): Estimated reward for each action.
        variance (float): Variance of the reward distribution.
        reward_distribution (Type[RewardDistribution]): Type of reward distribution used.
        rewards_history (List[List[float]]): History of rewards for each arm.
    """
    n_bandits: int
    optimistic_initialization: float
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    rewards_history: List[List[float]]

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
        )

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """
        Selects the action with the highest estimated reward.

        :returns: A tuple containing the index of the chosen action and the reward obtained from taking that action.
        :rtype: Tuple[int, float]
        """
        chosen_action_index = np.argmax(self.actions_estimated_reward)
        return chosen_action_index, self._update(chosen_action_index)

    def __repr__(self) -> str:
        return f"{super().__repr__()}(opt_init={self.optimistic_initialization})"

    def __str__(self):
        return f"""{self.__class__.__name__}(
                    n_bandits={self.n_bandits}\n
                    optimistic_initialization={self.optimistic_initialization})\n
                    Q_values={self.Q_values}\n
                    total_reward={self.current_step}\n
                    times_selected={self.times_selected}\n
                    actions_estimated_reward={self.actions_estimated_reward}\n
                    variance={self.variance}"""
