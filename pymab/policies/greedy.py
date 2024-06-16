import numpy as np
from typing import Tuple

from pymab.policies.policy import Policy
from pymab.reward_distribution import RewardDistribution


class GreedyPolicy(Policy):
    n_bandits: int
    optimistic_initilization: int
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: RewardDistribution

    def __init__(
        self,
        n_bandits: int,
        optimistic_initilization: int = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> None:
        super().__init__(
            n_bandits, optimistic_initilization, variance, reward_distribution
        )

    def select_action(self) -> Tuple[int, float]:
        chosen_action_index = np.argmax(self.actions_estimated_reward)
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
