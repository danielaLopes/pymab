"""Shared public types for PyMAB."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class RewardDomain(Enum):
    """Mathematical support required by an environment or policy."""

    REAL = "real"
    UNIT_INTERVAL = "unit_interval"
    BINARY = "binary"


class PolicyObjective(Enum):
    """Primary objective optimized by a policy."""

    CUMULATIVE_REWARD = "cumulative_reward"
    BEST_ARM = "best_arm"


@dataclass(frozen=True)
class PolicyCapabilities:
    """Static compatibility information for a policy implementation."""

    contextual: bool
    reward_domains: frozenset[RewardDomain]
    objective: PolicyObjective = PolicyObjective.CUMULATIVE_REWARD


ALL_REWARD_DOMAINS = frozenset(RewardDomain)


__all__ = [
    "ALL_REWARD_DOMAINS",
    "BoolArray",
    "FloatArray",
    "IntArray",
    "PolicyCapabilities",
    "PolicyObjective",
    "RewardDomain",
]
