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

    Args:
        n_bandits: Number of bandits (actions) available.
        context_dim: Dimension of the context vector.
        context_func: Function that generates context vectors.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution. Defaults to "gaussian".
        learning_rate: Rate at which linear coefficients are updated. Defaults to 0.1.

    Attributes:
        context_dim (int): Dimension of the context vector.
        theta (np.ndarray): Matrix of shape (n_bandits, context_dim) storing linear 
            coefficients for each bandit.
        learning_rate (float): Learning rate for updating coefficients.

    Note:
        Theory:
        Contextual Bandits extend traditional multi-armed bandits by incorporating
        contextual information. The policy learns a linear mapping from context to
        expected rewards for each action, enabling more informed decisions based on
        the current state or environment.

        The linear model updates follow the rule:
        θ_i = θ_i + α × (r - θ_i · c) × c
        where:
        - θ_i is the coefficient vector for action i
        - α is the learning rate
        - r is the observed reward
        - c is the context vector

    Example:
        A typical use case might be a recommendation system where the context
        includes user features:
        ```python
        def get_user_context():
            return np.array([
                [user.age, user.location, user.interests],  # features for action 1
                [user.age, user.location, user.interests],  # features for action 2
            ]).T

        policy = ContextualBanditPolicy(
            n_bandits=2,
            context_dim=3,
            context_func=get_user_context,
            learning_rate=0.1
        )
        ```
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

        Args:
            chosen_action_index: Index of the chosen action.
            context_chosen_action: Context vector when the action was chosen.

        Returns:
            The reward obtained from the chosen action.
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

        Args:
            context: Current context matrix of shape (context_dim, n_bandits).

        Returns:
            A tuple containing:
                - Index of the chosen action (int)
                - Reward received (float)

        Raises:
            ValueError: If context dimensions don't match expected dimensions.
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
