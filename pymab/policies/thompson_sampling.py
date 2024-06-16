import numpy as np
import math
from typing import Tuple

from pymab.policies.policy import Policy


import logging

from pymab.reward_distribution import RewardDistribution

logger = logging.getLogger(__name__)


class BernoulliThompsonSamplingPolicy(Policy):
    """
    https://www.youtube.com/watch?v=nkyDGGQ5h60
    https://towardsdatascience.com/multi-armed-bandits-thompson-sampling-algorithm-fea205cf31df
    """

    n_bandits: int
    optimistic_initilization: int
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: RewardDistribution

    times_success: np.array  # alpha
    times_failure: np.array  # beta

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
        self.times_success = np.zeros(self.n_bandits)
        self.times_failure = np.zeros(self.n_bandits)

    def _update(self, chosen_action_index: int) -> float:
        reward = super()._update(chosen_action_index)
        actual_mean = self._Q_values[chosen_action_index]
        # max_reward_action = np.argmax(self._Q_values)
        # TODO: Can we do this?? Since we shouldn't have access to this knowledge? what is the right way to do this?
        # if chosen_action_index == max_reward_action:
        # See how to determine success or failure here: https://visualstudiomagazine.com/articles/2019/06/01/thompson-sampling.aspx
        # if reward < actual_mean:
        logger.debug(f"reward {reward}")
        if reward < self._Q_values[chosen_action_index]:
            # self.times_success[chosen_action_index] += 1
            self.times_success[chosen_action_index] += 1
        else:
            # self.times_failure[chosen_action_index] += 1
            self.times_failure[chosen_action_index] += 1

        logger.debug(
            f"\nAction {chosen_action_index} was selected. Successes: {self.times_success[chosen_action_index]}, Failures: {self.times_failure[chosen_action_index]}"
        )
        logger.debug(f"Q Values {self._Q_values}")
        logger.debug(f"self.times_success {self.times_success}")
        logger.debug(f"self.times_failure {self.times_failure}")

        return reward

    def select_action(self) -> Tuple[int, float]:
        self.thomson_sampled = [
            np.random.beta(self.times_success[i] + 1, self.times_failure[i] + 1)
            for i in range(self.n_bandits)
        ]
        chosen_action_index = np.argmax(self.thomson_sampled)
        logger.debug(f"-------- self.thomson_sampled {self.thomson_sampled}")
        logger.debug(
            f"-------- self.actions_estimated_reward {self.actions_estimated_reward}"
        )
        logger.debug(f"chosen_action_index {chosen_action_index}")

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


class GaussianThompsonSamplingPolicy(Policy):
    """
    https://www.youtube.com/watch?v=nkyDGGQ5h60
    https://towardsdatascience.com/multi-armed-bandits-thompson-sampling-algorithm-fea205cf31df
    """

    n_bandits: int
    optimistic_initilization: int
    _Q_values: np.array
    current_step: int
    total_reward: float
    times_selected: np.array
    actions_estimated_reward: np.array
    variance: float
    reward_distribution: RewardDistribution

    means: np.array
    precisions: np.array

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
        self.means = np.zeros(n_bandits)
        self.precisions = np.ones(n_bandits) / variance

    def _update(self, chosen_action_index: int) -> float:
        reward = super()._update(chosen_action_index)

        prior_mean = self.means[chosen_action_index]
        prior_precision = self.precisions[chosen_action_index]

        posterior_mean = (prior_precision * prior_mean + reward) / (prior_precision + 1)
        posterior_precision = prior_precision + 1

        self.means[chosen_action_index] = posterior_mean
        self.precisions[chosen_action_index] = posterior_precision

        return reward

    def select_action(self) -> Tuple[int, float]:
        samples = [
            np.random.normal(self.means[i], 1 / np.sqrt(self.precisions[i]))
            for i in range(self.n_bandits)
        ]
        chosen_action_index = np.argmax(samples)

        return chosen_action_index, self._update(chosen_action_index)

    def __repr__(self) -> str:
        return f"{super().__repr__()}()"

    def __str__(self):
        return f"""{super().__repr__()}(
                    n_bandits={self.n_bandits}\n
                    Q_values={self.Q_values}\n
                    variance={self.variance}\n
                    means={self.means}\n
                    precisions={self.precisions}\n"""


class ThompsonSamplingPolicy:
    def __new__(
        cls,
        n_bandits: int,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> Policy:
        if reward_distribution == "bernoulli":
            return BernoulliThompsonSamplingPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
            )
        elif reward_distribution == "gaussian":
            return GaussianThompsonSamplingPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
            )
        else:
            raise ValueError(
                f"The {reward_distribution} distribution cannot be used with the Thomson Sampling policy!"
            )
