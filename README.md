# PyMAB

PyMAB is a typed Python library for reliable, reproducible multi-armed bandit
experiments. Version 2 separates environment, policy, decision, and observation
randomness; treats replicates as the independent unit of analysis; and records
enough provenance to reproduce saved results.

## Install

```bash
pip install pymab
```

Optional features are available through `pymab[plot]`, `pymab[analysis]`, and
`pymab[bayes]`.

## Basic experiment

```python
import numpy as np

from pymab import BanditEnvironment, Experiment, ExperimentConfig
from pymab.policies import EpsilonGreedyPolicy, UCBPolicy

result = Experiment(
    environment=BanditEnvironment(means=np.array([0.1, 0.4, 0.8])),
    policies={
        "epsilon-greedy": EpsilonGreedyPolicy(n_arms=3, epsilon=0.1),
        "ucb": UCBPolicy(n_arms=3),
    },
    config=ExperimentConfig(horizon=100, n_replicates=20, seed=42),
).run()

print(result.average_reward_by_step[-1])
print(result.cumulative_regret[-1])
```

Policy IDs are explicit and stable. Adding or reordering policies does not
change another policy's random stream or the simulated environment path.

## Paired comparison

```python
from pymab import compare
from pymab.policies import RandomPolicy, UCBPolicy

benchmark = compare(
    {"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)},
    environment=BanditEnvironment(means=np.array([0.1, 0.3, 0.8])),
    config=ExperimentConfig(horizon=100, n_replicates=20, seed=7),
    baseline="random",
    analysis_seed=91,
)

print(benchmark.lowest_mean_regret_policy)
print(benchmark.compare_to_baseline())
```

Intervals are deterministic paired bootstrap intervals over independent
replicates. A lower point estimate is not presented as statistical proof of a
winner.

## Offline evaluation

Fixed-policy estimators require logged-action propensities and report both raw
and post-clipping overlap diagnostics:

```python
from pymab import LoggedBanditDataset, estimate_policy_value


class FixedTarget:
    def probabilities(self, context):
        return np.array([0.25, 0.75])


logged = LoggedBanditDataset(
    actions=np.array([0, 1, 0, 1]),
    rewards=np.array([0.0, 1.0, 0.0, 1.0]),
    propensities=np.full(4, 0.5),
    n_arms=2,
)
estimate = estimate_policy_value(
    logged,
    FixedTarget(),
    method="snips",
    bootstrap_resamples=500,
    seed=7,
)
assert estimate.estimate == 0.75
```

Adaptive replay requires the logging design explicitly. Use
``logging_scheme="uniform"`` only for uniformly randomized logs; non-uniform
logs also require propensities and use rejection sampling.

## Probability environments

Additive Gaussian drift is intentionally rejected for Bernoulli means. Use
log-odds drift instead:

```python
from pymab import BernoulliReward, ProbabilityDrift

environment = BanditEnvironment(
    means=np.array([0.2, 0.5, 0.8]),
    reward_model=BernoulliReward(),
    dynamics=ProbabilityDrift(logit_std=0.05),
)
```

## Development

```bash
make sync
make format
make lint
make security
make test
make docs
make docs-linkcheck  # external network check; run separately
```

``make docs`` performs a clean warnings-as-errors HTML build, executes Sphinx
doctests, enforces 100% API docstring coverage, and runs every Python snippet
in this README.

The complete v1-to-v2 migration is documented in `docs/source/migration_v2.rst`.
