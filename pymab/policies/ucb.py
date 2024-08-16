from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import math
import typing

import numpy as np

from pymab.policies.mixins.stationarity_mixins import StationaryMixin, SlidingWindowMixin, DiscountedMixin
from pymab.policies.policy import Policy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class UCBPolicy(Policy, ABC):
    n_bandits: int
    optimistic_initialization: int
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
        optimistic_initialization: int = 0,
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
        pass

    def _get_ucb_value(self, action_index: int) -> float:
        if self.times_selected[action_index] == 0:
            return float("inf")

        mean_reward = self.actions_estimated_reward[action_index]

        return mean_reward + self._calculate_confidence_interval(action_index)

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
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


class StationaryUCBPolicy(StationaryMixin, UCBPolicy):
    n_bandits: int
    optimistic_initialization: int
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: Type[RewardDistribution]
    c: float

    def _calculate_confidence_interval(self, action_index: int) -> float:
        confidence_interval = math.sqrt(
                (self.c * math.log(self.current_step + 1))
                / self.times_selected[action_index]
            )
        return confidence_interval


class SlidingWindowUCBPolicy(SlidingWindowMixin, UCBPolicy):
    n_bandits: int
    optimistic_initialization: int
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
            optimistic_initialization: float = 0,
            variance: float = 1.0,
            reward_distribution: str = "gaussian",
            c: float = 1.0,
            window_size: int = 100
    ):
        UCBPolicy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c
        )
        SlidingWindowMixin.__init__(self, window_size=window_size)

    def _calculate_confidence_interval(self, action_index: int) -> float:
        confidence_interval = math.sqrt(
            (self.c * math.log(min(self.current_step, self.window_size) + 1))
            / min(self.times_selected[action_index], self.window_size)
        )
        return confidence_interval

    def __repr__(self) -> str:
        return f"{super().__repr__()}(sliding_window_size={self.window_size})"

    def __str__(self):
        description = super().__str__()
        return f"""{self.__class__.__name__}(
            {description}\n
            window_size={self.window_size})"""

class DiscountedUCBPolicy(DiscountedMixin, UCBPolicy):
    n_bandits: int
    optimistic_initialization: int
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
            discount_factor: float = 0.9
    ):
        UCBPolicy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c
        )
        DiscountedMixin.__init__(self, discount_factor=discount_factor)
        self.effective_n = 1 / (1 - self.discount_factor)

    def _calculate_confidence_interval(self, action_index: int) -> float:
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