"""Compare policies in a drifting environment."""

import numpy as np

from pymab.environments import BanditEnvironment, GradualDrift
from pymab.policies import DiscountedUCBPolicy, SlidingWindowUCBPolicy, UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig

environment = BanditEnvironment(
    means=np.array([0.2, 0.4, 0.6, 0.8]),
    dynamics=GradualDrift(std=0.01),
)

result = Experiment(
    environment=environment,
    policies={
        "ucb": UCBPolicy(n_arms=4),
        "sliding-window": SlidingWindowUCBPolicy(n_arms=4, window_size=50),
        "discounted": DiscountedUCBPolicy(n_arms=4, discount_factor=0.95),
    },
    config=ExperimentConfig(n_replicates=100, horizon=300, seed=7),
).run()

print(result.policy_ids)
print(result.optimal_action_rate_by_step[-1])
