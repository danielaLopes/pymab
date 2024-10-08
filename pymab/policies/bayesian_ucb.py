from __future__ import annotations

import logging
import typing

import numpy as np
import pymc3 as pm

from pymab.policies.ucb import UCBPolicy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class BernoulliBayesianUCBPolicy(UCBPolicy):
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
    n_mcmc_samples: int

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
        n_mcmc_samples: int = 1000,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c
        )
        self.n_mcmc_samples = n_mcmc_samples
        self.successes = np.zeros(n_bandits)
        self.failures = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        # Perform MCMC sampling to estimate the posterior distribution
        with pm.Model() as model:
            alpha = self.successes[action_index] + 1
            beta = self.failures[action_index] + 1
            theta = pm.Beta("theta", alpha=alpha, beta=beta)
            trace = pm.sample(self.n_mcmc_samples, chains=1, progressbar=False)

        # Get the upper quantile from the posterior distribution
        quantile_value = np.percentile(
            trace["theta"], 100 * (1 - 1 / (self.current_step + 1))
        )
        return quantile_value

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """


        Args:
            chosen_action_index (int): Index of the chosen action.

        Returns:
            float: Observed reward.
        """
        reward = super()._update(chosen_action_index)

        self.successes[chosen_action_index] += reward
        self.failures[chosen_action_index] += 1 - reward

        return reward

    # def select_action(self) -> Tuple[int, float]:
    #     if self.current_step < self.n_bandits:
    #         chosen_action_index = self.current_step
    #     else:
    #         ucb_values = np.zeros(self.n_bandits)
    #         for action_index in range(self.n_bandits):
    #             ucb_values[action_index] = self._get_ucb_value(action_index)
    #         chosen_action_index = np.argmax(ucb_values)
    #
    #     return chosen_action_index, self._update(chosen_action_index)

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
                    variance={self.variance}\n
                    c={self.c}\n
                    successes={self.successes}\n
                    failures={self.failures})"""


class GaussianBayesianUCBPolicy(UCBPolicy):
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
    n_mcmc_samples: int

    def __init__(
        self,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
        n_mcmc_samples: int = 1000,
    ) -> None:
        super().__init__(
            n_bandits, optimistic_initialization, variance, reward_distribution, c
        )
        self.n_mcmc_samples = n_mcmc_samples
        self.sum_rewards = np.zeros(n_bandits)
        self.sum_squared_rewards = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        # Perform MCMC sampling to estimate the posterior distribution
        with pm.Model() as model:
            mean = self.sum_rewards[action_index] / (
                self.times_selected[action_index] + 1
            )
            variance = (
                self.sum_squared_rewards[action_index]
                / (self.times_selected[action_index] + 1)
                - mean**2
            )
            theta = pm.Normal("theta", mu=mean, sigma=np.sqrt(variance))
            trace = pm.sample(self.n_mcmc_samples, chains=1, progressbar=False)

        quantile_value = np.percentile(
            trace["theta"], 100 * (1 - 1 / (self.current_step + 1))
        )
        return quantile_value

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """


        Args:
            chosen_action_index (int): Index of the chosen action.

        Returns:
            float: Observed reward.
        """
        reward = super()._update(chosen_action_index)

        self.sum_rewards[chosen_action_index] += reward
        self.sum_squared_rewards[chosen_action_index] += reward**2

        return reward

    # def select_action(self) -> Tuple[int, float]:
    #     if self.current_step < self.n_bandits:
    #         chosen_action_index = self.current_step
    #     else:
    #         ucb_values = np.array([self._get_ucb_value(i) for i in range(self.n_bandits)])
    #         chosen_action_index = np.argmax(ucb_values)
    #
    #     return chosen_action_index, self._update(chosen_action_index)

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
                    variance={self.variance}\n
                    c={self.c}\n
                    sum_rewards={self.sum_rewards}\n
                    sum_squared_rewards={self.sum_squared_rewards})"""


class BayesianUCBPolicy:
    def __new__(
        cls,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
        n_mcmc_samples: int = 1000,
    ) -> Union[BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy]:
        if reward_distribution == "bernoulli":
            return BernoulliBayesianUCBPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
                c=c,
                n_mcmc_samples=n_mcmc_samples,
            )
        elif reward_distribution == "gaussian":
            return GaussianBayesianUCBPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
                c=c,
                n_mcmc_samples=n_mcmc_samples,
            )
        else:
            raise ValueError(
                f"The {reward_distribution} distribution cannot be used with the Bayesian UCB policy!"
            )
