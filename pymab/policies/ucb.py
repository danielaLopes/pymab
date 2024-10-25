from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import math
import typing

import numpy as np

from pymab.policies.mixins.stationarity_mixins import (
    StationaryPolicyMixin,
    SlidingWindowMixin,
    DiscountedMixin,
)
from pymab.policies.policy import Policy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class UCBPolicy(StationaryPolicyMixin, Policy, ABC):
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
    c: float

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
        )
        self.c = c

    @abstractmethod
    def _calculate_confidence_interval(self, action_index: int) -> float:
        """
        Calculate the confidence interval for a given action.

        This method is abstract and should be implemented by subclasses to define
        the specific confidence interval calculation for each UCB variant.

        :param action_index: The index of the action to calculate the confidence interval for.
        :type action_index: int
        :return: The calculated confidence interval.
        :rtype: float
        """
        pass

    def _get_ucb_value(self, action_index: int) -> float:
        """
        Calculate the Upper Confidence Bound (UCB) value for a given action.

        This method implements the core UCB algorithm by combining the estimated reward
        with the confidence interval. It handles the case of unselected actions by
        returning infinity, ensuring exploration of all actions initially.

        :param action_index: The index of the action to calculate the UCB value for.
        :type action_index: int
        :return: The calculated UCB value, or infinity for unselected actions.
        :rtype: float

        :Theory:
            UCB = Q(a) + U(a), where Q(a) is the estimated reward and U(a) is the confidence interval.
        """
        if self.times_selected[action_index] == 0:
            return float("inf")

        mean_reward = self.actions_estimated_reward[action_index]

        return mean_reward + self._calculate_confidence_interval(action_index)

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """
        Select the next action based on the UCB algorithm.

        This method implements the action selection strategy of UCB:
        1. Initially, it selects each action once to gather initial estimates.
        2. After that, it chooses the action with the highest UCB value.

        :return: A tuple containing the index of the chosen action and the reward obtained from taking that action.
        :rtype: Tuple[int, float]

        Examples:
        Here's how to use the `select_action` method:

        .. code-block:: python

            policy = UCBPolicy(n_bandits=3)
            for _ in range(100):
                action, reward = policy.select_action()
                # Use the action and reward as needed
        """
        if self.current_step < self.n_bandits:
            chosen_action_index = self.current_step
        else:
            ucb_values = np.array(
                [self._get_ucb_value(i) for i in range(self.n_bandits)]
            )
            chosen_action_index = np.argmax(ucb_values)

        return chosen_action_index, self._update(chosen_action_index)

    def __repr__(self) -> str:
        return f"{super().__repr__()}(opt_init={self.optimistic_initialization}, c={self.c})"

    def __str__(self):
        return f"""{self.__class__.__name__}(
                    n_bandits={self.n_bandits}\n
                    optimistic_initialization={self.optimistic_initialization})\n
                    Q_values={self.Q_values}\n
                    total_reward={self.current_step}\n
                    times_selected={self.times_selected}\n
                    actions_estimated_reward={self.actions_estimated_reward}\n
                    variance={self.variance}
                    c={self.c})"""


class StationaryUCBPolicy(UCBPolicy):
    n_bandits: int
    optimistic_initialization: float
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    c: float

    def _calculate_confidence_interval(self, action_index: int) -> float:
        """
        Calculate the confidence interval for the stationary UCB algorithm.

        This method implements the standard UCB1 confidence interval calculation,
        which assumes a stationary environment (i.e., reward distributions do not change over time).

        :param action_index: The index of the action to calculate the confidence interval for.
        :type action_index: int
        :return: The calculated confidence interval.
        :rtype: float

        :Theory:
            The confidence interval is calculated as sqrt((c * log(t)) / n_a),
            where c is the exploration parameter, t is the current time step,
            and n_a is the number of times the action has been selected.

        :Optimization:
            - Uses math.sqrt and math.log for efficient calculation.
            - Adds 1 to current_step to avoid log(0) in the first step.
        """
        confidence_interval = math.sqrt(
            (self.c * math.log(self.current_step + 1))
            / self.times_selected[action_index]
        )
        return confidence_interval


class SlidingWindowUCBPolicy(SlidingWindowMixin, UCBPolicy):
    n_bandits: int
    optimistic_initialization: float
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    c: float
    window_size: int
    rewards_history: List[List[float]]

    def __init__(
        self,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
        window_size: int = 100,
    ):
        UCBPolicy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c,
        )
        SlidingWindowMixin.__init__(self, window_size=window_size)

    def _calculate_confidence_interval(self, action_index: int) -> float:
        """
        Calculate the confidence interval for the Sliding Window UCB algorithm.

        This method adapts the UCB confidence interval calculation to use a sliding window,
        which allows the algorithm to adapt to non-stationary environments by focusing on
        recent observations.

        :param action_index: The index of the action to calculate the confidence interval for.
        :type action_index: int
        :return: The calculated confidence interval.
        :rtype: float

        :Theory:
            The confidence interval is calculated similarly to UCB1, but uses the minimum of
            the current step (or window size) and the number of times the action has been selected
            within the window.

        :Optimization:
            - Uses min() to handle both the window size and the number of selections efficiently.
            - Adds 1 to the log term to avoid log(0) in the first step.
        """
        confidence_interval = math.sqrt(
            (self.c * math.log(min(self.current_step, self.window_size) + 1))
            / min(self.times_selected[action_index], self.window_size)
        )
        return confidence_interval

    def __repr__(self) -> str:
        return f"{super().__repr__()}(window={self.window_size})"

    def __str__(self):
        description = super().__str__()
        return f"""{self.__class__.__name__}(
            {description}\n
            window_size={self.window_size})"""


class DiscountedUCBPolicy(DiscountedMixin, UCBPolicy):
    n_bandits: int
    optimistic_initialization: float
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    c: float
    discount_factor: float
    effective_n: float

    def __init__(
        self,
        n_bandits: int,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
        discount_factor: float = 0.9,
    ):
        UCBPolicy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c,
        )
        DiscountedMixin.__init__(self, discount_factor=discount_factor)
        self.effective_n = 1 / (1 - self.discount_factor)

    def _calculate_confidence_interval(self, action_index: int) -> float:
        """
        Calculate the confidence interval for the Discounted UCB algorithm.

        This method implements the confidence interval calculation for Discounted UCB,
        which uses a discount factor to give more weight to recent observations,
        allowing adaptation to slowly varying non-stationary environments.

        :param action_index: The index of the action to calculate the confidence interval for.
        :type action_index: int
        :return: The calculated confidence interval.
        :rtype: float

        :Theory:
            The confidence interval is calculated using the effective sample size (effective_n)
            instead of the current time step. The effective_n is determined by the discount factor
            and represents the equivalent number of observations if all had full weight.

        :Optimization:
            - Precomputes effective_n in the constructor to avoid repeated calculations.
            - Uses min() to handle both the effective_n and the number of selections efficiently.
        """
        confidence_interval = math.sqrt(
            (self.c * math.log(self.effective_n))
            / min(self.times_selected[action_index], self.effective_n)
        )
        return confidence_interval

    def __repr__(self) -> str:
        return f"{super().__repr__()}(disc_f={self.discount_factor}, effect_n={round(self.effective_n, 2)})"

    def __str__(self):
        description = super().__str__()
        return f"""{self.__class__.__name__}(
            {description}\n
            discount_factor={self.discount_factor}\n
            effective_n={round(self.effective_n, 2)})"""
