import numpy as np
import random
from typing import Tuple 

from pymab.policies.greedy import GreedyPolicy


class EpsilonGreedyPolicy(GreedyPolicy):
    epsilon: float

    def __init__(self, 
                 n_bandits: int,
                 optimistic_initilization:int=0, 
                 variance: float=1.0,
                 epsilon:float=0.1) -> None:
        super().__init__(n_bandits, 
                         optimistic_initilization,
                         variance)
        self.epsilon = epsilon

    def select_action(self) -> Tuple[int, float]:
        r = random.uniform(
            0, 1
        )  # Choose random value to simulate whether the agent chooses greedy or non greedy, according to epsilon
        chosen_action_index = np.argmax(
            self.actions_estimated_reward
        )  # Find index of highest value action
        column_indexes = list(range(0, self.n_bandits))

        # Epsilon: choose a random action
        if r < self.epsilon:
            #column_indexes.remove(chosen_action_index)
            chosen_action_index = random.choice(
                column_indexes
            )  # Choose a random action index that is not the greedy choice
            return chosen_action_index, self._update(chosen_action_index)      
        # Non epsilon: choose greedily
        else:
            return chosen_action_index, self._update(chosen_action_index)
                
    def __repr__(self):
        return f"{self.__class__.__name__}(opt_init={self.optimistic_initilization}, ε={self.epsilon})"