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

    :ivar n_bandits: Number of bandit arms
    :type n_bandits: int
    :ivar optimistic_initialization: Initial optimistic value for estimated rewards
    :type optimistic_initialization: float
    :ivar _Q_values: True Q-values for each arm (set externally)
    :type _Q_values: np.ndarray
    :ivar current_step: Current time step in the learning process
    :type current_step: int
    :ivar total_reward: Cumulative reward obtained so far
    :type total_reward: float
    :ivar times_selected: Number of times each arm has been selected
    :type times_selected: np.ndarray
    :ivar actions_estimated_reward: Estimated reward for each action
    :type actions_estimated_reward: np.ndarray
    :ivar variance: Variance of the reward distribution
    :type variance: float
    :ivar reward_distribution: Type of reward distribution used
    :type reward_distribution: Type[RewardDistribution]
    :ivar rewards_history: History of rewards for each arm
    :type rewards_history: List[List[float]]

    .. note::
        The Greedy policy is a simple but effective approach for multi-armed bandit problems.
        It always chooses the action that currently appears to be the best, based on the
        estimated rewards. This can lead to quick convergence to a good solution, but may
        also get stuck in local optima if the initial estimates are inaccurate.

    .. note::
        Optimistic initialization can be used to encourage initial exploration by setting
        the initial estimated rewards higher than expected. This helps to ensure that all
        actions are tried at least once before settling on a preferred action.
    """

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

        This method implements the core of the Greedy algorithm by choosing the action
        with the highest estimated reward based on current knowledge.

        :return: A tuple containing the index of the chosen action and the reward obtained from taking that action
        :rtype: Tuple[int, float]

        .. note::
            The Greedy policy does not explicitly explore, which means it may miss out on
            potentially better actions if the initial estimates are inaccurate. This is known
            as the exploration-exploitation trade-off in reinforcement learning.
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
