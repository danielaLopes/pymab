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

    :ivar n_bandits: Number of bandits (actions) available
    :type n_bandits: int
    :ivar optimistic_initialization: Initial Q-value for all actions
    :type optimistic_initialization: float
    :ivar _Q_values: True Q-values for each arm (set externally)
    :type _Q_values: np.ndarray
    :ivar current_step: Current time step in the learning process
    :type current_step: int
    :ivar total_reward: Cumulative reward obtained so far
    :type total_reward: float
    :ivar times_selected: Number of times each action has been selected
    :type times_selected: np.ndarray
    :ivar actions_estimated_reward: Estimated reward for each action
    :type actions_estimated_reward: np.ndarray
    :ivar variance: Variance of the reward distribution
    :type variance: float
    :ivar reward_distribution: Type of reward distribution
    :type reward_distribution: Type[RewardDistribution]
    :ivar rewards_history: History of rewards for each action
    :type rewards_history: List[List[float]]
    :ivar c: Exploration parameter for UCB calculation
    :type c: float
    :ivar n_mcmc_samples: Number of MCMC samples (not used in current implementation)
    :type n_mcmc_samples: int
    :ivar alpha: Alpha parameters of the Beta distribution for each arm
    :type alpha: np.ndarray
    :ivar beta: Beta parameters of the Beta distribution for each arm
    :type beta: np.ndarray
    :ivar successes: Number of successful outcomes for each arm
    :type successes: np.ndarray
    :ivar failures: Number of failed outcomes for each arm
    :type failures: np.ndarray

    .. note::
        Theory:
        The Bernoulli Bayesian UCB policy maintains a Beta distribution for each arm,
        which represents the current belief about the probability of success. The policy
        selects actions based on an upper confidence bound of these distributions, balancing
        exploration and exploitation. The successes and failures for each arm are tracked
        separately to update the Beta distribution parameters.

    .. note::
        Optimizations:
        - Uses numpy arrays for efficient storage and computation of distribution parameters.
        - Implements optimistic initialization through the initial values of alpha and beta.
        - Tracks successes and failures separately for quick updates and probability calculations.
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
    n_mcmc_samples: int

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "bernoulli",
        c: float = 1.0,
        n_mcmc_samples: int = 1000,
    ) -> None:
        super().__init__(
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
            c=c,
        )
        self.n_mcmc_samples = n_mcmc_samples
        self.successes = np.zeros(n_bandits)
        self.failures = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        """
        Calculates the UCB value for a given action.

        This method computes the upper confidence bound using the properties of the Beta distribution.

        :param action_index: Index of the action to calculate UCB for.
        :type action_index: int

        :returns: The calculated UCB value.
        :rtype: float
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

        :param chosen_action_index: Index of the chosen action.
        :type chosen_action_index: int

        :returns: Observed reward.
        :rtype: float
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

    This policy uses a Normal-Inverse-Gamma distribution as a conjugate prior for Gaussian rewards.
    It calculates the UCB value using the mean and variance of the posterior distribution.

    :param n_bandits: Number of bandit arms.
    :type: int
    :param optimistic_initialization: Initial optimistic value for estimated rewards.
    :type: float
    :param variance: Variance of the reward distribution.
    :type: float
    :param c: Exploration parameter for UCB calculation.
    :type: float
    :param n_mcmc_samples: Number of MCMC samples (not used in current implementation).
    :type: int
    :param sum_rewards: Array to keep track of sum of rewards for each arm.
    :type: np.array
    :param sum_squared_rewards: Array to keep track of sum of squared rewards for each arm.
    :type: np.array
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
            n_bandits=n_bandits, optimistic_initialization=optimistic_initialization, variance=variance, reward_distribution=reward_distribution, c=c
        )
        self.n_mcmc_samples = n_mcmc_samples
        self.sum_rewards = np.zeros(n_bandits)
        self.sum_squared_rewards = np.zeros(n_bandits)

    def _get_ucb_value(self, action_index: int) -> float:
        """
        Calculates the UCB value for a given action.

        This method computes the upper confidence bound using the properties of the Normal distribution.

        :param action_index: Index of the action to calculate UCB for.
        :type: int

        :returns: The calculated UCB value.
        :rtype: float
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

        :param chosen_action_index: Index of the chosen action.
        :type: int

        :returns: Observed reward.
        :rtype: float
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

    :ivar n_bandits: Number of bandits (actions) available
    :type n_bandits: int
    :ivar optimistic_initialization: Initial Q-value for all actions
    :type optimistic_initialization: float
    :ivar variance: Variance of the reward distribution
    :type variance: float
    :ivar reward_distribution: Type of reward distribution
    :type reward_distribution: str
    :ivar c: Exploration parameter for UCB calculation
    :type c: float
    :ivar n_mcmc_samples: Number of MCMC samples (not used in current implementation)
    :type n_mcmc_samples: int

    .. note::
        Theory:
        Bayesian UCB policies use Bayesian inference to estimate the distribution of rewards
        for each action. This allows for a more principled approach to balancing exploration
        and exploitation, taking into account the uncertainty in the reward estimates.

    .. note::
        The choice between Bernoulli and Gaussian policies depends on the nature of the
        reward distribution in the problem being solved.
    """

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

