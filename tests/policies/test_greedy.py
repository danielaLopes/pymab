import unittest
import numpy as np
from pymab.policies.greedy import GreedyPolicy
from pymab.reward_distribution import (
    GaussianRewardDistribution,
    BernoulliRewardDistribution,
    UniformRewardDistribution,
)
from tests.policies.test_policy import TestPolicy


class TestOptimisticGreedyPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = GreedyPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 1.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
        )


class TestGreedyPolicy(TestPolicy):
    def setUp(self):
        self.policy_class = GreedyPolicy
        self.n_bandits = 3
        self.optimistic_initialization = 0.0
        self.variance = 1.0
        self.reward_distribution = "gaussian"
        self.policy = self.policy_class(
            n_bandits=self.n_bandits,
            optimistic_initialization=self.optimistic_initialization,
            variance=self.variance,
            reward_distribution=self.reward_distribution,
        )


if __name__ == "__main__":
    unittest.main()
