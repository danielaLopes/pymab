import numpy as np
import math
from typing import Tuple 

from pymab.policies.policy import Policy


class ThomsonSamplingPolicy(Policy):
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
    prior_mean: float
    prior_variance: float
    posterior_mean: float
    posterior_variance: float


    def __init__(self, 
                 n_bandits: int,
                 optimistic_initilization: int=0, 
                 variance: float=1.0, 
                 c: int=1) -> None:
        super().__init__(n_bandits,
                         optimistic_initilization,
                         variance)
        self.prior_mean = self.Q_values_mean
        self.prior_variance = self.Q_values_variance
        self.posterior_mean = self.Q_values_mean
        self.posterior_variance = self.Q_values_variance


    def _update_prior_posterior(self, reward: float) -> Tuple[float, float]:
        # TODO
        posterior_mean_num = self.Q_values_variance * reward + 
        posterior_mean_den = 
        posterior_mean = posterior_mean_num / posterior_mean_den
        return posterior_mean, posterior_variance


    def select_action(self) -> Tuple[int, float]:
        # TODO
        self._update_prior_posterior(reward)
        # if self.current_step < self.n_bandits:
        #     chosen_action_index = self.current_step
        # else:
        #     ucb_values = np.zeros(self.n_bandits)
        #     for action_index in range(0, self.n_bandits):
        #         if self.times_selected[action_index] > 0:
        #             ucb_values[action_index] = (
        #                 math.sqrt(self.c * 
        #                     math.log(self.current_step + 1)
        #                     / self.times_selected[action_index]
        #                 )
        #             )
        #     chosen_action_index = np.argmax(
        #         self.actions_estimated_reward + ucb_values
        #     )
        
        return chosen_action_index, self._update(chosen_action_index)
    

    def __repr__(self) -> str:
        return f"{super().__repr__()}(opt_init={self.optimistic_initilization}, c={self.c})"
    

    def __str__(self):
        return f"""{super().__repr__()}(
                    n_bandits={self.n_bandits}\n
                    optimistic_initilization={self.optimistic_initilization})\n
                    Q_values={self.Q_values}\n
                    total_reward={self.current_step}\n
                    times_selected={self.times_selected}\n
                    actions_estimated_reward={self.actions_estimated_reward}\n
                    variance={self.variance})"""