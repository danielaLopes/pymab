"""PyMAB: reliable multi-armed bandit experiments."""

from importlib.metadata import PackageNotFoundError, version

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
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult

try:
    __version__ = version("pymab")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

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
