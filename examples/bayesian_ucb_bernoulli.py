"""Compare Bernoulli UCB-style policies."""

import numpy as np

from pymab.distributions import BernoulliReward
from pymab.environments import BanditEnvironment
from pymab.policies import (
    BernoulliBayesianUCBPolicy,
    BernoulliThompsonSamplingPolicy,
    UCBPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig

environment = BanditEnvironment(
    means=np.array([0.15, 0.25, 0.6, 0.8]),
    reward_model=BernoulliReward(),
)

result = Experiment(
    environment=environment,
    policies={
        "ucb": UCBPolicy(n_arms=4),
        "bayesian-ucb": BernoulliBayesianUCBPolicy(n_arms=4),
        "thompson": BernoulliThompsonSamplingPolicy(n_arms=4),
    },
    config=ExperimentConfig(n_replicates=200, horizon=500, seed=42),
).run()

print(result.policy_ids)
print(result.cumulative_regret[-1])
