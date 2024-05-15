from abc import abstractmethod
import numpy as np
from typing import List, Tuple


class Policy:
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
        self.n_bandits = n_bandits
        self.optimistic_initilization = optimistic_initilization
        self._Q_values = None
        self.current_step = 0
        self.total_reward = 0
        self.variance = variance
        self.times_selected = np.zeros(self.n_bandits)
        self.actions_estimated_reward = np.full(self.n_bandits,  
                                                self.optimistic_initilization, 
                                                dtype=float)


    def _get_actual_reward(self, action_index: int) -> float:
        true_reward = np.random.normal(self.Q_values[action_index], self.variance)
        return true_reward


    def _update(self, chosen_action_index: int) -> float:
        self.current_step += 1
        reward = self._get_actual_reward(chosen_action_index)
        self.total_reward += reward
        self.times_selected[chosen_action_index] += 1
        # Calculate average reward per action without storing all rewards
        self.actions_estimated_reward[chosen_action_index] += (
            (reward - self.actions_estimated_reward[chosen_action_index]) /
            self.times_selected[chosen_action_index]
        )
        return reward
    

    @property
    def Q_values(self) -> List[float]:
        if self._Q_values is None:
            raise ValueError("Q_values not set yet!")
        return self._Q_values

    @Q_values.setter
    def Q_values(self, Q_values: List[float]) -> None:
        if len(Q_values) != self.n_bandits:
            raise ValueError("Q_values length needs to match n_bandits!")
        self._Q_values = Q_values

    @abstractmethod
    def select_action(self) -> Tuple[int, float]:
        pass


    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.times_selected = np.zeros(self.n_bandits)
        self.actions_estimated_reward = np.full(self.n_bandits,  
                                                self.optimistic_initilization, 
                                                dtype=float)


    def __repr__(self) -> str:
        return self.__class__.__name__