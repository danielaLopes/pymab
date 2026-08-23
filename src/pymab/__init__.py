"""PyMAB: reliable multi-armed bandit experiments."""

from pymab._version import __version__
from pymab.benchmarking import BenchmarkResult, compare
from pymab.distributions import (
    BernoulliReward,
    BetaArmPrior,
    GaussianArmPrior,
    GaussianReward,
    UniformArmPrior,
    UniformReward,
)
from pymab.environments import (
    AbruptShift,
    BanditEnvironment,
    ContextProvider,
    GradualDrift,
    LinearContextualEnvironment,
    LogisticContextualEnvironment,
    ProbabilityDrift,
    RandomArmSwap,
    StationaryDynamics,
)
from pymab.errors import (
    CompatibilityError,
    OverlapError,
    PyMABError,
    SerializationError,
    ValidationError,
)
from pymab.offline import (
    BatchTargetPolicy,
    CrossFittedRewardModel,
    LoggedBanditDataset,
    LoggingScheme,
    OfflineEstimate,
    SequentialReplayResult,
    TargetPolicy,
    estimate_policy_value,
    sequential_replay,
)
from pymab.provenance import RunProvenance
from pymab.results import SimulationResult
from pymab.simulation import Experiment, ExperimentConfig

__all__ = [
    "__version__",
    "AbruptShift",
    "BanditEnvironment",
    "BatchTargetPolicy",
    "BenchmarkResult",
    "BernoulliReward",
    "BetaArmPrior",
    "CompatibilityError",
    "ContextProvider",
    "CrossFittedRewardModel",
    "Experiment",
    "ExperimentConfig",
    "GaussianArmPrior",
    "GaussianReward",
    "GradualDrift",
    "LinearContextualEnvironment",
    "LoggedBanditDataset",
    "LoggingScheme",
    "LogisticContextualEnvironment",
    "OfflineEstimate",
    "OverlapError",
    "ProbabilityDrift",
    "PyMABError",
    "RandomArmSwap",
    "RunProvenance",
    "SequentialReplayResult",
    "SerializationError",
    "SimulationResult",
    "StationaryDynamics",
    "TargetPolicy",
    "UniformArmPrior",
    "UniformReward",
    "ValidationError",
    "compare",
    "estimate_policy_value",
    "sequential_replay",
]
