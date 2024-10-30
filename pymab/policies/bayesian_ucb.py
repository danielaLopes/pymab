from __future__ import annotations

import logging
import math
import typing

import numpy as np
from scipy import stats

from pymab.policies.ucb import StationaryUCBPolicy
from pymab.reward_distribution import RewardDistribution

if typing.TYPE_CHECKING:
    from typing import *

logger = logging.getLogger(__name__)


class BernoulliBayesianUCBPolicy(StationaryUCBPolicy):
    """
    Implements a Bayesian Upper Confidence Bound (UCB) policy for Bernoulli bandits.

    This policy uses Bayesian inference with a Beta prior to estimate the probability of success
    for each arm, and selects actions based on an upper confidence bound of these estimates.

    Args:
        n_bandits: Number of bandits (actions) available.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution. Must be "bernoulli". Defaults to "bernoulli".
        c: Exploration parameter for UCB calculation. Defaults to 1.0.

    Attributes:
        successes (np.ndarray): Number of successful outcomes for each arm.
        failures (np.ndarray): Number of failed outcomes for each arm.

    Note:
        Theory:
        The Bernoulli Bayesian UCB policy maintains a Beta distribution for each arm,
        which represents the current belief about the probability of success. The policy
        selects actions based on an upper confidence bound of these distributions, 
        balancing exploration and exploitation.

        Optimizations:
        - Uses numpy arrays for efficient storage and computation
        - Implements optimistic initialization through Beta parameters
        - Tracks successes and failures separately for quick updates
    """
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

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "bernoulli",
        c: float = 1.0,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c,
        )
        self.successes = np.zeros(n_bandits)
        self.failures = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        """
        Calculates the UCB value for a given action.

        This method computes the upper confidence bound using the properties of the Beta distribution.

        Args:
            action_index: Index of the action to calculate UCB for.

        Returns:
            The calculated UCB value incorporating uncertainty from the Beta distribution.

        Note:
            The UCB value is computed using the mean and variance of the Beta distribution,
            with the variance scaled by the exploration parameter and current step.
        """
        alpha = self.successes[action_index] + 1
        beta = self.failures[action_index] + 1

        mean = alpha / (alpha + beta)

        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        ucb = mean + self.c * math.sqrt(variance / (self.current_step + 1))

        return ucb

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Updates the policy after an action is taken.

        This method updates the success and failure counts based on the observed reward.

        Args:
            chosen_action_index: Index of the chosen action.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The observed reward.
        """
        reward = super()._update(chosen_action_index)

        self.successes[chosen_action_index] += reward
        self.failures[chosen_action_index] += 1 - reward

        return reward

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


class GaussianBayesianUCBPolicy(StationaryUCBPolicy):
    """
    Implements a Bayesian Upper Confidence Bound (UCB) policy for Gaussian-distributed rewards.

    This policy uses a Normal distribution as a conjugate prior for Gaussian rewards.
    It calculates the UCB value using the mean and variance of the posterior distribution.

    Args:
        n_bandits: Number of bandits (actions) available.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution. Must be "gaussian". Defaults to "gaussian".
        c: Exploration parameter for UCB calculation. Defaults to 1.0.

    Attributes:
        sum_rewards (np.ndarray): Sum of rewards for each arm.
        sum_squared_rewards (np.ndarray): Sum of squared rewards for each arm.

    Note:
        The policy maintains running sums of rewards and squared rewards to efficiently
        compute the posterior mean and variance for each arm. This allows for quick
        updates and UCB calculations without storing full reward histories.
    """
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

    def __init__(
        self,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits, optimistic_initialization=optimistic_initialization, variance=variance, reward_distribution=reward_distribution, c=c
        )
        self.sum_rewards = np.zeros(n_bandits)
        self.sum_squared_rewards = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        """
        Calculates the UCB value for a given action.

        This method computes the upper confidence bound using the properties of the Normal distribution.

        Args:
            action_index: Index of the action to calculate UCB for.

        Returns:
            The calculated UCB value incorporating uncertainty from the Normal distribution.

        Note:
            Returns infinity for unselected actions to ensure initial exploration.
            Uses the Normal distribution's properties to compute a confidence bound
            based on the empirical mean and variance.
        """
        if self.times_selected[action_index] == 0:
            return float('inf')

        mean = self.sum_rewards[action_index] / self.times_selected[action_index]

        variance = (
                self.sum_squared_rewards[action_index] / self.times_selected[action_index]
                - mean ** 2
        )

        variance = max(variance, 1e-10)

        standard_error = math.sqrt(variance / self.times_selected[action_index])

        confidence_level = 1 - 1 / (self.current_step + 1)
        z_score = stats.norm.ppf(confidence_level)
        ucb = mean + z_score * standard_error

        return ucb

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Updates the policy after an action is taken.

        This method updates the sum of rewards and sum of squared rewards based on the observed reward.

        Args:
            chosen_action_index: Index of the chosen action.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The observed reward.
        """
        reward = super()._update(chosen_action_index)

        self.sum_rewards[chosen_action_index] += reward
        self.sum_squared_rewards[chosen_action_index] += reward**2

        return reward

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
    """
    Factory class for creating Bayesian UCB policies based on the reward distribution.

    This class creates and returns either a BernoulliBayesianUCBPolicy or a GaussianBayesianUCBPolicy
    based on the specified reward distribution.

    Args:
        n_bandits: Number of bandits (actions) available.
        optimistic_initialization: Initial Q-value for all actions. Defaults to 0.0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution ("bernoulli" or "gaussian"). 
            Defaults to "gaussian".
        c: Exploration parameter for UCB calculation. Defaults to 1.0.

    Returns:
        Either a BernoulliBayesianUCBPolicy or GaussianBayesianUCBPolicy instance.

    Raises:
        ValueError: If an unsupported reward distribution is specified.

    Note:
        The choice between Bernoulli and Gaussian policies should be based on the
        nature of the rewards in the problem being solved. Bernoulli is suitable
        for binary outcomes, while Gaussian is better for continuous rewards.
    """

    def __new__(
        cls,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
        c: float = 1.0,
    ) -> Union[BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy]:
        if reward_distribution == "bernoulli":
            return BernoulliBayesianUCBPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
                c=c,
            )
        elif reward_distribution == "gaussian":
            return GaussianBayesianUCBPolicy(
                n_bandits=n_bandits,
                variance=variance,
                reward_distribution=reward_distribution,
                c=c,
            )
        else:
            raise ValueError(
                f"The {reward_distribution} distribution cannot be used with the Bayesian UCB policy!"
            )

