from __future__ import annotations

import numpy as np
import pytest

from pymab import BanditEnvironment, Experiment, ExperimentConfig
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.policies.policy import Policy
from pymab.results import SimulationResult


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_seeded_backend_replays_and_records_its_rng_scheme(backend: str) -> None:
    def run() -> SimulationResult:
        return Experiment(
            environment=BanditEnvironment(means=np.array([0.1, 0.9])),
            policies={
                "random": RandomPolicy(n_arms=2),
                "ucb": UCBPolicy(n_arms=2),
            },
            config=ExperimentConfig(
                horizon=10,
                n_replicates=3,
                seed=81,
                backend=backend,  # type: ignore[arg-type]
            ),
        ).run()

    left = run()
    right = run()
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    assert left.provenance.backend == backend
    assert left.provenance.rng_scheme != "unknown"


def test_native_added_policy_does_not_change_existing_trajectory() -> None:
    def run(include_ucb: bool) -> SimulationResult:
        policies: dict[str, Policy] = {"random": RandomPolicy(n_arms=2)}
        if include_ucb:
            policies["ucb"] = UCBPolicy(n_arms=2)
        return Experiment(
            environment=BanditEnvironment(means=np.array([0.1, 0.9])),
            policies=policies,
            config=ExperimentConfig(
                horizon=10,
                n_replicates=3,
                seed=81,
                backend="rust",
            ),
        ).run()

    alone = run(False)
    together = run(True)
    np.testing.assert_array_equal(alone.actions[:, :, 0], together.actions[:, :, 0])
    np.testing.assert_array_equal(alone.rewards[:, :, 0], together.rewards[:, :, 0])
