import numpy as np
from typing import Tuple

from pymab.policies.policy import Policy


class GreedyPolicy(Policy):
    n_bandits: int
    optimistic_initilization: int
    _Q_values: np.array
    Q_values_mean: float
    Q_values_variance: float
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float


    def __init__(self, 
                 n_bandits: int,
                 optimistic_initilization: int=0, 
                 variance: float=1.0) -> None:
        super().__init__(n_bandits,
                         optimistic_initilization,
                         variance)


    def select_action(self) -> Tuple[int, float]:
        chosen_action_index = np.argmax(
            self.actions_estimated_reward
        )
        return chosen_action_index, self._update(chosen_action_index)
    

    def __repr__(self) -> str:
        return f"{super().__repr__()}(opt_init={self.optimistic_initilization})"


    def __str__(self):
        return f"""{self.__class__.__name__}(
                    n_bandits={self.n_bandits}\n
                    optimistic_initilization={self.optimistic_initilization})\n
                    Q_values={self.Q_values}\n
                    total_reward={self.current_step}\n
                    times_selected={self.times_selected}\n
                    actions_estimated_reward={self.actions_estimated_reward}\n
                    variance={self.variance}"""