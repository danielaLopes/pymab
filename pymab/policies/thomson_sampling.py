import numpy as np
import math
from typing import Tuple

from pymab.policies.policy import Policy


import logging

logger = logging.getLogger(__name__)


class ThomsonSamplingPolicy(Policy):
    """
    https://www.youtube.com/watch?v=nkyDGGQ5h60
    """
    n_bandits: int
    optimistic_initilization: int
    _Q_values: np.array
    Q_values_mean: float # rewards mean
    Q_values_variance: float # reward variance
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float

    times_success: np.array # alpha
    times_failure: np.array # beta


    def __init__(self, 
                 n_bandits: int,
                 optimistic_initilization: int=0, 
                 variance: float=1.0,
                 Q_values_mean: float=0) -> None:
        super().__init__(n_bandits,
                         optimistic_initilization,
                         variance)
        self.times_success = np.zeros(self.n_bandits)
        self.times_failure = np.zeros(self.n_bandits)
        self.Q_values_mean = Q_values_mean


    def _update(self, chosen_action_index: int) -> float:
        reward = super()._update(chosen_action_index)
        actual_mean = self._Q_values[chosen_action_index]
        # max_reward_action = np.argmax(self._Q_values)
        # TODO: Can we do this?? Since we shouldn't have access to this knowledge? what is the right way to do this?
        #if chosen_action_index == max_reward_action:
        # See how to determine success or failure here: https://visualstudiomagazine.com/articles/2019/06/01/thompson-sampling.aspx
        # if reward < actual_mean:
        logger.debug(
            f"reward {reward}")
        if reward < self._Q_values[chosen_action_index]:
            # self.times_success[chosen_action_index] += 1
            self.times_success[chosen_action_index] += 1
        else:
            # self.times_failure[chosen_action_index] += 1
            self.times_failure[chosen_action_index] += 1

        logger.debug(f"\nAction {chosen_action_index} was selected. Successes: {self.times_success[chosen_action_index]}, Failures: {self.times_failure[chosen_action_index]}")
        logger.debug(
            f"Q Values {self._Q_values}")
        logger.debug(
            f"self.times_success {self.times_success}")
        logger.debug(
            f"self.times_failure {self.times_failure}")

        return reward

    def select_action(self) -> Tuple[int, float]:
        self.thomson_sampled = [np.random.beta(self.times_success[i] + 1, self.times_failure[i] + 1) for i in range(self.n_bandits)]
        chosen_action_index = np.argmax(
            self.thomson_sampled
        )
        logger.debug(
            f"-------- self.thomson_sampled {self.thomson_sampled}")
        logger.debug(
            f"-------- self.actions_estimated_reward {self.actions_estimated_reward}")
        logger.debug(
            f"chosen_action_index {chosen_action_index}")
        
        return chosen_action_index, self._update(chosen_action_index)
    

    def __repr__(self) -> str:
        return f"{super().__repr__()}()"
    

    def __str__(self):
        return f"""{super().__repr__()}(
                    n_bandits={self.n_bandits}\n
                    Q_values={self.Q_values}\n
                    variance={self.variance}\n
                    times_success={self.times_success}\n
                    times_failure={self.times_failure}\n"""