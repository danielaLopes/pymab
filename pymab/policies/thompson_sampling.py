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

    Args:
        n_bandits: Number of bandit arms available.
        optimistic_initialization: Initial value for all action estimates. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Must be "bernoulli". Defaults to "bernoulli".

    Attributes:
        successes (np.ndarray): Number of successful outcomes for each arm.
        failures (np.ndarray): Number of failed outcomes for each arm.
        thomson_sampled (List[float]): Last sampled values from posterior distributions.

    Note:
        Theory:
        Thompson Sampling implements Bayesian exploration by:
        1. Maintaining Beta(α, β) posterior for each arm
        2. α = successes + 1, β = failures + 1 (adding 1 for uniform prior)
        3. Sampling θ ~ Beta(α, β) for each arm
        4. Selecting arm with highest sampled θ
        
        The Beta distribution is the conjugate prior for Bernoulli likelihood,
        making updates simple and computationally efficient.

    Example:
        ```python
        policy = BernoulliThompsonSamplingPolicy(
            n_bandits=5,
            reward_distribution="bernoulli"
        )

        # Run for 1000 steps
        for _ in range(1000):
            action, reward = policy.select_action()
            # Process binary reward (0 or 1)...
        ```
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
            chosen_action_index: Index of the chosen action.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The observed reward (0 or 1).

        Note:
            Uses conjugate prior property of Beta-Bernoulli:
            - Success (reward = 1): Increment α
            - Failure (reward = 0): Increment β
        """
        reward = super()._update(chosen_action_index)

        if reward > 0:
            self.successes[chosen_action_index] += 1
        else:
            self.failures[chosen_action_index] += 1

        return reward

    def select_action(self, *args, **kwargs) -> Tuple[int, float]:
        """Select action using Thompson Sampling.

        Returns:
            A tuple containing:
                - Index of the chosen action (int)
                - Reward received for the action (float)

        Note:
            Implementation:
            1. Sample θ ~ Beta(α, β) for each arm
            2. Select arm with highest θ
            3. Observe reward and update parameters
        """
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

    Args:
        n_bandits: Number of bandit arms available.
        optimistic_initialization: Initial value for all action estimates. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Must be "gaussian". Defaults to "gaussian".

    Attributes:
        means (np.ndarray): Posterior mean for each arm.
        precisions (np.ndarray): Posterior precision (1/variance) for each arm.

    Note:
        Theory:
        The policy maintains a Normal distribution for each arm's mean reward:
        1. Uses conjugate Normal prior with known variance
        2. Updates posterior mean and precision after each observation
        3. Samples from posterior and selects highest sample
        
        The Normal distribution is conjugate to itself with known variance,
        allowing closed-form Bayesian updates.

    Example:
        ```python
        policy = GaussianThompsonSamplingPolicy(
            n_bandits=5,
            variance=1.0,
            reward_distribution="gaussian"
        )

        # Run for 1000 steps
        for _ in range(1000):
            action, reward = policy.select_action()
            # Process continuous reward...
        ```
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

        Args:
            chosen_action_index: Index of the chosen action.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The observed reward (continuous value).

        Note:
            Implementation:
            1. Compute posterior mean using precision-weighted average
            2. Update precision by adding 1 (for unit variance likelihood)
            3. Store updated parameters for chosen arm
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
        """Select action using Thompson Sampling.

        Returns:
            A tuple containing:
                - Index of the chosen action (int)
                - Reward received for the action (float)

        Note:
            Implementation:
            1. Sample μ ~ N(mean, 1/precision) for each arm
            2. Select arm with highest sampled μ
            3. Observe reward and update parameters
        """
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
    """Factory class for creating Thompson Sampling policies.

    This class serves as a factory to create the appropriate Thompson Sampling policy
    based on the reward distribution type (Bernoulli or Gaussian).

    Args:
        n_bandits: Number of bandit arms available.
        optimistic_initialization: Initial value for all action estimates. Defaults to 0.
        variance: Variance of the reward distribution. Defaults to 1.0.
        reward_distribution: Type of reward distribution ("bernoulli" or "gaussian").
            Defaults to "gaussian".

    Returns:
        The appropriate Thompson Sampling policy instance.

    Raises:
        ValueError: If an unsupported reward distribution is specified.

    Example:
        ```python
        # Create a Bernoulli policy
        policy = ThompsonSamplingPolicy(
            n_bandits=5,
            reward_distribution="bernoulli"
        )

        # Create a Gaussian policy
        policy = ThompsonSamplingPolicy(
            n_bandits=5,
            reward_distribution="gaussian"
        )
        ```
    """
    
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
