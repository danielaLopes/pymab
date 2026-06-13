# PyMAB

PyMAB is a small Python library for reproducible multi-armed bandit experiments.
It separates environments, policies, simulations, metrics, and plotting so
experiments are easier to test, compare, and extend.

## What changed in v1

- Python 3.11+ and a clean typed API.
- Explicit random seeds through `numpy.random.Generator`.
- Environments own true arm values and reward sampling.
- Policies only select actions and update from observed rewards.
- Regret is computed from expected rewards, with realized rewards kept separate.
- Plotting dependencies are optional via `pymab[plot]`.

## Install

```bash
pip install pymab
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
python -m unittest discover -v
```

With uv, use the Makefile targets used by CI:

```bash
make sync
make format
make lint
make security
make test
```

`make format` checks formatting. Use `make format-fix` to rewrite formatting
locally.

The Makefile tries `uv run ...` first. If your local uv binary fails before it
can run a tool, the targets fall back to executables installed in `.venv`; run
`make sync` once to create/populate that environment.

## Basic Example

```python
import numpy as np

from pymab.environments import BanditEnvironment
from pymab.policies import EpsilonGreedyPolicy, UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig

environment = BanditEnvironment(q_values=np.array([0.1, 0.4, 0.8]))
policies = [
    EpsilonGreedyPolicy(n_arms=3, epsilon=0.1),
    UCBPolicy(n_arms=3, c=2.0),
]

result = Experiment(
    environment=environment,
    policies=policies,
    config=ExperimentConfig(n_episodes=200, n_steps=500, seed=42),
).run()

print(result.average_reward_by_step[-1])
print(result.cumulative_regret[-1])
```

## Non-Stationary Environments

```python
import numpy as np

from pymab.environments import BanditEnvironment, GradualDrift

environment = BanditEnvironment(
    q_values=np.array([0.2, 0.5, 0.7]),
    dynamics=GradualDrift(change_rate=0.01),
)
```

Built-in dynamics:

- `StationaryDynamics`
- `GradualDrift`
- `AbruptShift`
- `RandomArmSwap`

## Policies

Classic bandits:

- `GreedyPolicy`
- `EpsilonGreedyPolicy`
- `SoftmaxPolicy`
- `GradientBanditPolicy`
- `UCBPolicy`
- `SlidingWindowUCBPolicy`
- `DiscountedUCBPolicy`
- `BernoulliThompsonSamplingPolicy`
- `GaussianThompsonSamplingPolicy`
- `BernoulliBayesianUCBPolicy`
- `GaussianBayesianUCBPolicy`

Contextual bandits:

- `LinearEpsilonGreedyPolicy`
- `LinUCBPolicy`
- `LinearThompsonSamplingPolicy`

## Contextual Example

```python
import numpy as np

from pymab.environments import LinearContextualEnvironment
from pymab.policies import LinUCBPolicy
from pymab.simulation import Experiment, ExperimentConfig


def context_provider(rng: np.random.Generator) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


environment = LinearContextualEnvironment(
    theta=np.array([[1.0, 0.0], [0.0, 1.0]]),
    context_provider=context_provider,
)

result = Experiment(
    environment=environment,
    policies=[LinUCBPolicy(n_arms=2, n_features=2)],
    config=ExperimentConfig(n_episodes=100, n_steps=200, seed=7),
).run()
```

## Plotting

```python
from pathlib import Path

from pymab.plotting import plot_average_reward, plot_cumulative_regret

plot_average_reward(result, output_path=Path("results/average_reward.html"))
plot_cumulative_regret(result, output_path=Path("results/regret.html"))
```

Install plotting extras first:

```bash
pip install "pymab[plot]"
```

## Migration Notes

The old `Game` API remains as a deprecated compatibility wrapper. New code
should use `BanditEnvironment`, `ExperimentConfig`, and `Experiment` directly.
The old `pymab.reward_distribution` import path also remains available, but the
new module is `pymab.distributions`.
