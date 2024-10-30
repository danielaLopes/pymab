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

    Args:
        n_bandits: Number of bandits (actions) available.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution. Defaults to "gaussian".
        epsilon: Exploration rate (0 ≤ ε ≤ 1). Defaults to 0.1.

    Attributes:
        epsilon (float): Probability of choosing a random action for exploration.

    Note:
        Theory:
        The Epsilon-Greedy strategy addresses the exploration-exploitation dilemma
        by using a simple probability-based approach:
        
        1. Exploration (ε probability):
           - Randomly select any non-greedy action
           - Helps discover potentially better actions
           - Prevents getting stuck in local optima
        
        2. Exploitation (1-ε probability):
           - Select the action with highest estimated reward
           - Capitalizes on current knowledge
           - Maximizes immediate reward

        The value of ε controls the balance:
        - Higher ε: More exploration, slower convergence
        - Lower ε: More exploitation, risk of suboptimal solutions

    Example:
        ```python
        # Create a policy with 10% exploration rate
        policy = EpsilonGreedyPolicy(
            n_bandits=5,
            epsilon=0.1,
            reward_distribution="gaussian"
        )
        
        # Select actions
        for _ in range(100):
            action, reward = policy.select_action()
            # Process reward...
        ```
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

        Returns:
            A tuple containing:
                - Index of the chosen action (int)
                - Reward received for the action (float)

        Note:
            Implementation Details:
            1. Generate random number r ∈ [0, 1]
            2. If r < ε:
               - Find the greedy action (highest estimated reward)
               - Remove it from possible choices
               - Select randomly from remaining actions
            3. If r ≥ ε:
               - Select the greedy action (highest estimated reward)
            
            This ensures that exploration explicitly avoids the greedy action,
            promoting true exploration of alternative actions.
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
