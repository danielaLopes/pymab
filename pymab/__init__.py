"""PyMAB: reproducible multi-armed bandit experiments."""

from pymab.distributions import BernoulliReward, GaussianReward, UniformReward
from pymab.environments import (
    AbruptShift,
    BanditEnvironment,
    EnvironmentChangeType,
    GradualDrift,
    LinearContextualEnvironment,
    RandomArmSwap,
    StationaryDynamics,
)
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AbruptShift",
    "BanditEnvironment",
    "BernoulliReward",
    "EnvironmentChangeType",
    "Experiment",
    "ExperimentConfig",
    "GaussianReward",
    "GradualDrift",
    "LinearContextualEnvironment",
    "RandomArmSwap",
    "SimulationResult",
    "StationaryDynamics",
    "UniformReward",
]
