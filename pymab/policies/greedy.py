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

    Args:
        n_bandits: Number of bandit arms available.
        optimistic_initialization: Initial value for all action estimates. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution. Defaults to "gaussian".

    Attributes:
        Inherits all attributes from Policy class, with no additional attributes.

    Note:
        Theory:
        The Greedy policy represents the pure exploitation approach to the
        exploration-exploitation dilemma:

        1. Action Selection:
           - Always choose arg max_a Q(a)
           - Q(a) is the estimated reward for action a
           - No explicit exploration mechanism

        2. Key Characteristics:
           - Fast convergence when initial estimates are accurate
           - Risk of suboptimal performance with poor initialization
           - No guaranteed exploration of all actions

        3. Optimistic Initialization:
           Can be used to encourage initial exploration by setting high
           initial values, causing the policy to try actions until their
           estimates drop below the best-known action's value.

    Example:
        ```python
        # Create a policy with optimistic initialization
        policy = GreedyPolicy(
            n_bandits=5,
            optimistic_initialization=1.0,
            reward_distribution="gaussian"
        )

        # Run for 100 steps
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

        Returns:
            A tuple containing:
                - Index of the chosen action (int)
                - Reward received for the action (float)

        Note:
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
