"""Compare Bernoulli UCB-style policies."""

import numpy as np

from pymab.distributions import BernoulliReward
from pymab.environments import BanditEnvironment
from pymab.policies import BayesianUCBPolicy, ThompsonSamplingPolicy, UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig

environment = BanditEnvironment(
    q_values=np.array([0.15, 0.25, 0.6, 0.8]),
    reward_distribution=BernoulliReward(),
)

result = Experiment(
    environment=environment,
    policies=[
        UCBPolicy(n_arms=4),
        BayesianUCBPolicy(n_arms=4, reward_distribution="bernoulli"),
        ThompsonSamplingPolicy(n_arms=4, reward_distribution="bernoulli"),
    ],
    config=ExperimentConfig(n_episodes=200, n_steps=500, seed=42),
).run()

print(result.policy_names)
print(result.cumulative_regret[-1])
