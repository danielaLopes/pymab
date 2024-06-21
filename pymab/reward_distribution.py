from abc import ABC, abstractmethod

import numpy as np


class RewardDistribution(ABC):
    """Abstract base class for reward distributions.
    This class defines the interface that all reward distribution classes must implement.
    """

    @staticmethod
    @abstractmethod
    def get_reward(q_value: float, variance: float) -> float:
        """Abstract method to get a reward based on the distribution.

        Args:
            q_value (float): The mean or central value of the distribution.
            variance (float): The variance or spread of the distribution.

        Returns:
            float: A reward sampled from the distribution.
        """
        pass


class GaussianRewardDistribution(RewardDistribution):
    """Gaussian (Normal) reward distribution."""

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        """Get a reward sampled from a Gaussian distribution.

        Args:
            q_value (float): The mean of the Gaussian distribution.
            variance (float): The standard deviation of the Gaussian distribution.

        Returns:
            float: A reward sampled from the Gaussian distribution.

        Example:
            If q_value = 0.3 and variance = 1, it is likely that the sampled value will be between -0.7 and 1.3, and most will be around 0.3.
        """
        return np.random.normal(q_value, variance)


class BernoulliRewardDistribution(RewardDistribution):
    """Bernoulli reward distribution."""

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        """Get a reward sampled from a Bernoulli distribution.

        Args:
            q_value (float): The probability of success in the Bernoulli distribution.
            variance (float): Not used in Bernoulli distribution, included for interface compatibility.

        Returns:
            float: A reward sampled from the Bernoulli distribution.

        Example:
            If q_value = 0.3, approximately 70% of the sampled values will be 1 (successes) and 30% will be 0 (unsuccesses).
        """
        return np.random.binomial(1, q_value, size=1)


class UniformRewardDistribution(RewardDistribution):
    """Uniform reward distribution."""

    @staticmethod
    def get_reward(q_value: float, variance: float) -> float:
        """Get a reward sampled from a Uniform distribution.

        Args:
            q_value (float): The central value around which the uniform distribution is centered.
            variance (float): The half-range of the uniform distribution.

        Returns:
            float: A reward sampled from the uniform distribution.

        Example:
            If q_value = 0.3 and variance = 1, every value between -0.7 and 1.3 has equal probability of being sampled.
        """
        return np.random.uniform(q_value - variance, q_value + variance)
