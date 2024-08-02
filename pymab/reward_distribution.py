from __future__ import annotations

from abc import ABC, abstractmethod
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from typing import *


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

    @staticmethod
    @abstractmethod
    def generate_Q_values(
        q_value: float, variance: float, size: int
    ) -> np.ndarray[float]:
        """Abstract method to get a set of Q values.

        Args:
            q_value (float): The mean or central value of the distribution.
            variance (float): The variance or spread of the distribution.
            size (int): The number of Q values to generate.

        Returns:
            np.ndarray[float]: A list of Q-values sampled from the distribution.
        """
        pass

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__.__name__)


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
            If q_value = 0.3 and variance = 1, approximately 68% of the data will be between -0.7 and 1.3
            (mean ± variance), approximately 95% of the data will be between -1.7 and 2.3 (mean ± 2 * variance),
            and approximately 99.7% of the data will be between -2.7 and 3.3 (mean ± 3 * variance).
        """
        return np.random.normal(q_value, variance)

    @staticmethod
    def generate_Q_values(
        q_value: float, variance: float, size: int
    ) -> np.ndarray[float]:
        """Get a set of Q values sampled from a Gaussian distribution.

        Args:
            q_value (float): The mean or central value of the distribution.
            variance (float): The variance or spread of the distribution.
            size (int): The number of Q values to generate.

        Returns:
            np.ndarray[float]: A list of Q-values sampled from the distribution.
        """
        return np.random.normal(q_value, variance, size)


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

    @staticmethod
    def generate_Q_values(
        q_value: float, variance: float, size: int
    ) -> np.ndarray[float]:
        """Get a set of Q values sampled from a Bernoulli distribution.

        Args:
            q_value (float): The mean or central value of the distribution.
            variance (float): The variance or spread of the distribution.
            size (int): The number of Q values to generate.

        Returns:
            np.ndarray[float]: A list of Q-values sampled from the distribution.
        """
        return np.random.binomial(1, q_value, size)


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

    @staticmethod
    def generate_Q_values(
        q_value: float, variance: float, size: int
    ) -> np.ndarray[float]:
        """Get a set of Q values sampled from a Uniform distribution.

        Args:
            q_value (float): The mean or central value of the distribution.
            variance (float): The variance or spread of the distribution.
            size (int): The number of Q values to generate.

        Returns:
            np.ndarray[float]: A list of Q-values sampled from the distribution.
        """
        return np.random.uniform(q_value - variance, q_value + variance, size)
