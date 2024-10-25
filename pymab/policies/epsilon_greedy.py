from __future__ import annotations

import logging

import numpy as np
import random
import typing

from pymab.policies.greedy import GreedyPolicy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class EpsilonGreedyPolicy(GreedyPolicy):
    """
    Implements the Epsilon-Greedy policy for multi-armed bandit problems.

    This policy balances exploration and exploitation by choosing the best-known action
    (exploitation) with probability 1-ε, and a random action (exploration) with probability ε.

    :ivar n_bandits: Number of bandits (actions) available
    :type n_bandits: int
    :ivar optimistic_initialization: Initial Q-value for all actions
    :type optimistic_initialization: float
    :ivar _Q_values: True Q-values for each arm (set externally)
    :type _Q_values: np.ndarray
    :ivar current_step: Current time step in the learning process
    :type current_step: int
    :ivar total_reward: Cumulative reward obtained so far
    :type total_reward: float
    :ivar times_selected: Number of times each action has been selected
    :type times_selected: np.ndarray
    :ivar actions_estimated_reward: Estimated reward for each action
    :type actions_estimated_reward: np.ndarray
    :ivar variance: Variance of the reward distribution
    :type variance: float
    :ivar reward_distribution: Type of reward distribution
    :type reward_distribution: Type[RewardDistribution]
    :ivar rewards_history: History of rewards for each action
    :type rewards_history: List[List[float]]
    :ivar epsilon: Exploration rate
    :type epsilon: float

    .. note::
        Theory:
        The Epsilon-Greedy policy addresses the exploration-exploitation dilemma in reinforcement
        learning. It improves upon the purely greedy approach by allowing for occasional
        exploration, which can lead to discovering better actions in the long run.

    .. note::
        Optimizations:
        - Uses numpy arrays for efficient storage and computation of Q-values and action counts.
        - Implements optimistic initialization to encourage initial exploration.
    """

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
        )
        self.epsilon = epsilon

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """
        Select an action based on the Epsilon-Greedy policy.

        This method implements the core of the Epsilon-Greedy algorithm:
        - With probability ε, choose a random action (exploration).
        - With probability 1-ε, choose the action with the highest estimated reward (exploitation).

        :return: A tuple containing the index of the chosen action and the updated reward estimate for the chosen action
        :rtype: Tuple[int, float]

        .. note::
            The method ensures that during exploration, the greedy action is not selected,
            forcing true exploration.
        """
        r = random.uniform(0, 1)
        chosen_action_index = np.argmax(self.actions_estimated_reward)
        
        if r < self.epsilon:  # Explore
            column_indexes = list(range(0, self.n_bandits))
            column_indexes.pop(chosen_action_index)
            chosen_action_index = random.choice(column_indexes)
        # else: Exploit (chosen_action_index is already set to the greedy choice)

        return chosen_action_index, self._update(chosen_action_index)

    def __repr__(self):
        return f"{self.__class__.__name__}(opt_init={self.optimistic_initialization}, ε={self.epsilon})"
