"""Backward-compatible exports for reward distributions.

The v1 API lives in :mod:`pymab.distributions`. This module keeps the old
import path available while avoiding the previous global-random implementation.
"""

from pymab.distributions import (
    BernoulliReward,
    GaussianReward,
    RewardDistribution,
    UniformReward,
)

GaussianRewardDistribution = GaussianReward
BernoulliRewardDistribution = BernoulliReward
UniformRewardDistribution = UniformReward

__all__ = [
    "RewardDistribution",
    "GaussianReward",
    "BernoulliReward",
    "UniformReward",
    "GaussianRewardDistribution",
    "BernoulliRewardDistribution",
    "UniformRewardDistribution",
]
