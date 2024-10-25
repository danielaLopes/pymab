from __future__ import annotations

import numpy as np
import typing

from matplotlib import pyplot as plt
from scipy.stats import beta, norm

from pymab.policies.mixins.stationarity_mixins import StationaryPolicyMixin
from pymab.policies.policy import Policy


import logging

from pymab.reward_distribution import RewardDistribution

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from typing import *


class BernoulliThompsonSamplingPolicy(StationaryPolicyMixin, Policy):
    """
    This policy is used for multi-armed bandit problems with Bernoulli-distributed rewards.
    It uses the Beta distribution to model the probability of success for each action and
    updates these probabilities based on observed rewards.

    :ivar n_bandits: Number of bandit arms.
    :type n_bandits: int
    :ivar optimistic_initialization: Initial optimistic value for estimated rewards.
    :type optimistic_initialization: float
    :ivar _Q_values: True values of the bandit arms.
    :type _Q_values: np.array
    :ivar current_step: Current step count.
    :type current_step: int
    :ivar total_reward: Total accumulated reward.
    :type total_reward: float
    :ivar times_selected: Count of times each bandit arm has been selected.
    :type times_selected: np.array
    :ivar actions_estimated_reward: Estimated rewards for each action.
    :type actions_estimated_reward: np.array
    :ivar variance: Variance of the reward distribution.
    :type variance: float
    :ivar reward_distribution: Type of reward distribution used. Should always be Bernoulli.
    :type reward_distribution: RewardDistribution
    :ivar successes: Alpha values for Beta distribution (success counts).
    :type successes: np.array
    :ivar failures: Beta values for Beta distribution (failure counts).
    :type failures: np.array
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
    rewards_history: List[List[float]]
    successes: np.array
    failures: np.array

    def __init__(
        self,
        *,
        n_bandits: int,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> None:
        Policy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
        )
        self.successes = np.zeros(self.n_bandits)
        self.failures = np.zeros(self.n_bandits)

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Updates the parameters successes and failures used in the Beta distribution, according to the reward
        obtained.
        The Bernoulli distribution is conjugate to the Beta distribution, meaning that if the prior distribution of the
        probability of success is a Beta distribution, then the posterior distribution after observing data is also a
        Beta distribution. This makes the Bayesian updating process straightforward.

        Args:
            chosen_action_index (int): Index of the chosen action.

        :return: Observed reward.
        :rtype: float
        """
        reward = super()._update(chosen_action_index)

        if reward > 0:
            self.successes[chosen_action_index] += 1
        else:
            self.failures[chosen_action_index] += 1

        return reward

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        self.thomson_sampled = [
            np.random.beta(self.successes[i] + 1, self.failures[i] + 1)
            for i in range(self.n_bandits)
        ]
        chosen_action_index = np.argmax(self.thomson_sampled)

        return chosen_action_index, self._update(chosen_action_index)

    def __repr__(self) -> str:
        return f"{super().__repr__()}()"

    def __str__(self):
        return f"""{super().__repr__()}(
                    n_bandits={self.n_bandits}\n
                    Q_values={self.Q_values}\n
                    variance={self.variance}\n
                    successes={self.successes}\n
                    failures={self.failures}\n"""

    def plot_distribution(self) -> None:
        """
        Plots the distributions of the expected reward for the current step.

        :param policy: The policy instance (either BernoulliThompsonSamplingPolicy or GaussianThompsonSamplingPolicy).
        :type policy: Union[BernoulliThompsonSamplingPolicy, GaussianThompsonSamplingPolicy]
        :param step_num: The number of steps (iterations) the policy has been executed.
        :type step_num: int
        """
        fig, axes = plt.subplots(
            1, self.n_bandits, figsize=(15, 6), constrained_layout=True
        )
        x_range = (
            np.linspace(0, 1, 1000)
            if isinstance(self, BernoulliThompsonSamplingPolicy)
            else np.linspace(-2, 2, 1000)
        )

        for i in range(self.n_bandits):
            a, b = self.successes[i] + 1, self.failures[i] + 1
            y = beta.pdf(x_range, a, b)
            axes[i].plot(x_range, y, label=f"Arm {i} Posterior")
            axes[i].axvline(
                self.Q_values[i], color="r", linestyle="--", label="True Reward"
            )

            axes[i].legend()
            axes[i].set_title(f"Arm {i} - Step {self.current_step}")
            axes[i].text(
                0.5,
                -0.1,
                f"Successes: {self.successes[i]}, Failures: {self.failures[i]}",
                transform=axes[i].transAxes,
                ha="center",
                va="top",
            )

        fig.suptitle(f"Reward distribution for {self.__class__.__name__}")
        plt.show()


class GaussianThompsonSamplingPolicy(StationaryPolicyMixin, Policy):
    """
    This policy is used for multi-armed bandit problems with Gaussian-distributed rewards.
    It models the mean reward for each action using a Gaussian distribution and updates
    these means based on observed rewards.

    :ivar n_bandits: Number of bandit arms.
    :type n_bandits: int
    :ivar optimistic_initialization: Initial optimistic value for estimated rewards.
    :type optimistic_initialization: float
    :ivar _Q_values: True values of the bandit arms.
    :type _Q_values: np.array
    :ivar current_step: Current step count.
    :type current_step: int
    :ivar total_reward: Total accumulated reward.
    :type total_reward: float
    :ivar times_selected: Count of times each bandit arm has been selected.
    :type times_selected: np.array
    :ivar actions_estimated_reward: Estimated rewards for each action.
    :type actions_estimated_reward: np.array
    :ivar variance: Variance of the reward distribution.
    :type variance: float
    :ivar reward_distribution: Type of reward distribution used. Should always be Gaussian.
    :type reward_distribution: RewardDistribution
    :ivar means: Mean rewards for each action.
    :type means: np.array
    :ivar precisions: Precision (inverse of variance) for each action.
    :type precisions: np.array
    """

    n_bandits: int
    optimistic_initialization: float
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
        optimistic_initialization: float = 0.0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> None:
        Policy.__init__(
            self,
            n_bandits=n_bandits,
            optimistic_initialization=optimistic_initialization,
            variance=variance,
            reward_distribution=reward_distribution,
        )
        self.means = np.zeros(n_bandits)
        self.precisions = np.ones(n_bandits) / variance

    def _update(self, chosen_action_index: int, *args, **kwargs) -> float:
        """
        Updates the Guassian prior distribution according to the observed reward. The conjugate prior for the mean of a Gaussian distribution with known variance is also Gaussian.
            The posterior distribution of the mean given Gaussian observations remains Gaussian, which allows for a Bayesian update, but it involves maintaining and updating the mean and variance parameters.
        The means and precisions (tau) arrays maintain the posterior mean and precision (inverse of variance) for each action. These are updated after each observed reward using Bayesian inference.

        :param chosen_action_index: Index of the chosen action.
        :type chosen_action_index: int

        :return: Observed reward.
        :rtype: float
        """
        reward = super()._update(chosen_action_index)

        prior_mean = self.means[chosen_action_index]
        prior_precision = self.precisions[chosen_action_index]

        posterior_mean = (prior_precision * prior_mean + reward) / (prior_precision + 1)
        posterior_precision = prior_precision + 1

        self.means[chosen_action_index] = posterior_mean
        self.precisions[chosen_action_index] = posterior_precision

        return reward

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
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

    def plot_distribution(self) -> None:
        """
        Plots the distributions of the expected reward for the current step.

        :param policy: The policy instance (either BernoulliThompsonSamplingPolicy or GaussianThompsonSamplingPolicy).
        :type policy: Union[BernoulliThompsonSamplingPolicy, GaussianThompsonSamplingPolicy]
        :param step_num: The number of steps (iterations) the policy has been executed.
        :type step_num: int
        """
        fig, axes = plt.subplots(
            1, self.n_bandits, figsize=(15, 6), constrained_layout=True
        )
        x_range = (
            np.linspace(0, 1, 1000)
            if isinstance(self, BernoulliThompsonSamplingPolicy)
            else np.linspace(-2, 2, 1000)
        )

        for i in range(self.n_bandits):
            mean, precision = self.means[i], self.precisions[i]
            std_dev = 1 / np.sqrt(precision)
            y = norm.pdf(x_range, mean, std_dev)
            axes[i].plot(x_range, y, label=f"Arm {i} Posterior")
            axes[i].axvline(
                self.Q_values[i], color="r", linestyle="--", label="True Reward"
            )

            axes[i].legend()
            axes[i].set_title(f"Arm {i} - Step {self.current_step}")
            axes[i].text(
                0.5,
                -0.1,
                f"Mean: {round(self.means[i], 2)}, Precisions: {round(self.precisions[i], 2)}",
                transform=axes[i].transAxes,
                ha="center",
                va="top",
            )

        fig.suptitle(f"Reward distribution for {self.__class__.__name__}")
        plt.show()


class ThompsonSamplingPolicy:
    def __new__(
        cls,
        n_bandits: int,
        optimistic_initialization: float = 0,
        variance: float = 1.0,
        reward_distribution: str = "gaussian",
    ) -> Union[BernoulliThompsonSamplingPolicy, GaussianThompsonSamplingPolicy]:
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
