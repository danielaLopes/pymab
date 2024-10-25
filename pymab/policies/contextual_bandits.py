from __future__ import annotations

import logging

import numpy as np
import typing

from pymab.policies.mixins.stationarity_mixins import StationaryPolicyMixin
from pymab.policies.policy import Policy

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class ContextualBanditPolicy(StationaryPolicyMixin, Policy):
    """
    Implements a Contextual Bandit policy for multi-armed bandit problems.

    This policy introduces the notion that the reward obtained at each step depends on the current context,
    such as the time of the day, or whether it's raining or not. When deciding which action to take, the agent leverages
    its context to make a more informed decision. This potentially reduces the need for exploration, unlike other
    policies.

    Contextual Bandits implements a linear model to predict rewards based on the context, where the weights are updated.

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
    :ivar context_dim: The dimension of the context
    :type context_dim: int
    :ivar theta: Matrix of shape (n_bandits, context_dim) where each row stores the linear coefficients for each bandit
    :type theta: np.ndarray
    :ivar learning_rate: Controls how quickly the linear coefficients (theta) are updated based on new observations
    :type learning_rate: float

    .. note::
        Theory:
        Contextual Bandits extend the traditional multi-armed bandit problem by incorporating
        contextual information. This allows the policy to adapt its decisions based on the
        current state or context, potentially leading to more efficient learning and better
        performance in non-stationary environments.

    .. note::
        Optimizations:
        - Uses numpy arrays for efficient storage and computation of theta values.
        - Implements a linear model for quick updates and predictions.
    """

    def __init__(
        self,
        *,
        n_bandits: int,
        context_dim: int,
        context_func: Callable,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        learning_rate: float = 0.1,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            context_func=context_func,
        )
        self.context_dim = context_dim
        self.theta = np.zeros(
            (n_bandits, context_dim)
        )
        self.learning_rate = learning_rate

    def _update(
        self, chosen_action_index: int, context_chosen_action: np.array
    ) -> float:
        """
        Updates the linear coefficients for the chosen action.
        The coefficients are updated using the following formula:
        θ_i = θ_i + α × (r - θ_i · c) × c, where c is the context and α is the learning rate.

        :param chosen_action_index: The index of the chosen action
        :type chosen_action_index: int
        :param context_chosen_action: The context when the action was chosen
        :type context_chosen_action: np.array
        :return: The reward obtained from the chosen action
        :rtype: float
        """
        reward = super()._update(chosen_action_index)
        self.theta[chosen_action_index] += (
            self.learning_rate
            * (reward - self.theta[chosen_action_index] @ context_chosen_action)
            * context_chosen_action
        )

        return reward

    def reset(self) -> None:
        """
        Reset the policy to its initial state.

        This method resets the base policy and reinitializes the theta matrix.
        """
        super().reset()
        self.theta = np.zeros((self.n_bandits, self.context_dim))

    def select_action(self, context: np.array) -> Tuple[int, float]:
        """
        Selects the action based on the current context using a linear model, leveraging the dot product between thetas
        and the current context, which represents the weighted sum of the features in the current context, and used to
        estimate rewards.

        :param context: The current context. It's shape should be (context_dim, n_bandits), inverse of theta.
        :type: np.array
        :returns: The selected action and the expected reward.
        :rtype: Tuple[int, float]
        :raises ValueError: If the context dimensions do not match the expected dimensions
        """
        if context.shape[0] != self.context_dim:
            raise ValueError(
                "Context dimension does not match the expected context_dim."
            )
        if context.shape[1] != self.n_bandits:
            raise ValueError("Context dimension does not match the expected n_bandits.")

        expected_rewards = np.array(
            [self.theta[i] @ context[:, i] for i in range(self.n_bandits)]
        )
        chosen_action_index = np.argmax(expected_rewards)

        return chosen_action_index, self._update(
            chosen_action_index, context_chosen_action=context[:, chosen_action_index]
        )

    def __repr__(self) -> str:
        return f"{super().__repr__()}(learning_rate={self.learning_rate})"

    def __str__(self):
        return f"""{super().__repr__()}(
                    n_bandits={self.n_bandits}\n
                    Q_values={self.Q_values}\n
                    variance={self.variance}\n
                    context_dim={self.context_dim}\n,
                    theta={self.theta}\n,
                    learning_rate={self.learning_rate})"""
