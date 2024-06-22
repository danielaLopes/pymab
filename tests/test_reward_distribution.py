import unittest
import numpy as np
from pymab.reward_distribution import (
    GaussianRewardDistribution,
    BernoulliRewardDistribution,
    UniformRewardDistribution,
)


class TestGaussianRewardDistribution(unittest.TestCase):
    def test_get_reward(self):
        """
        Test the get_reward method of the GaussianRewardDistribution class.
        Due to the properties of the Gaussian distribution, the reward should be within 3 standard deviations of the mean.
        Sets see for reproducibility.
        """
        q_value = 0.3
        variance = 1.0
        np.random.seed(0)
        reward = GaussianRewardDistribution.get_reward(q_value, variance)
        self.assertTrue(
            q_value - 3 * variance <= reward <= q_value + 3 * variance,
            "Reward is not within the expected range.",
        )


class TestBernoulliRewardDistribution(unittest.TestCase):
    def test_get_reward(self):
        """
        Test the get_reward method of the BernoulliRewardDistribution class.
        Variance is not used in the Bernoulli distribution.
        Sets seed for reproducibility.
        """
        q_value = 0.3
        variance = 1.0
        np.random.seed(0)
        reward = BernoulliRewardDistribution.get_reward(q_value, variance)
        self.assertIn(reward, [0, 1], "Reward is not 0 or 1.")
        self.assertTrue(
            isinstance(reward, np.ndarray) and reward.shape == (1,),
            "Reward should be a numpy array of shape (1,)",
        )


class TestUniformRewardDistribution(unittest.TestCase):
    def test_get_reward(self):
        """
        Test the get_reward method of the UniformRewardDistribution class.
        The reward should be within the range of q_value ± variance.
        Sets seed for reproducibility.
        """
        q_value = 0.3
        variance = 1.0
        np.random.seed(0)
        reward = UniformRewardDistribution.get_reward(q_value, variance)
        self.assertTrue(
            q_value - variance <= reward <= q_value + variance,
            "Reward is not within the expected range.",
        )


if __name__ == "__main__":
    unittest.main()
