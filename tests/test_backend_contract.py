from __future__ import annotations

from dataclasses import fields

import numpy as np

from benchmarks.cases import CASE_NAMES, build_experiment
from benchmarks.reference_worker import measure_case
from pymab import BanditEnvironment, Experiment, ExperimentConfig
from pymab.policies import EpsilonGreedyPolicy, UCBPolicy


def test_reference_experiment_config_contract() -> None:
    assert tuple(field.name for field in fields(ExperimentConfig)) == (
        "horizon",
        "n_replicates",
        "seed",
        "reward_coupling",
        "record_contexts",
    )


def test_reference_result_array_and_provenance_contract() -> None:
    result = Experiment(
        environment=BanditEnvironment(means=np.array([0.1, 0.5, 0.9])),
        policies={
            "epsilon": EpsilonGreedyPolicy(n_arms=3, epsilon=0.1),
            "ucb": UCBPolicy(n_arms=3),
        },
        config=ExperimentConfig(horizon=4, n_replicates=2, seed=11),
    ).run()

    assert result.rewards.shape == (2, 4, 2)
    assert result.actions.shape == (2, 4, 2)
    assert result.expected_rewards.shape == (2, 4, 2)
    assert result.recommendations.shape == (2, 4, 2)
    assert result.arm_means.shape == (2, 4, 3)
    assert result.optimal_mask.shape == (2, 4, 3)
    assert result.contexts is None
    assert result.rewards.dtype == np.dtype(np.float64)
    assert result.actions.dtype == np.dtype(np.int64)
    assert result.expected_rewards.dtype == np.dtype(np.float64)
    assert result.recommendations.dtype == np.dtype(np.int64)
    assert result.arm_means.dtype == np.dtype(np.float64)
    assert result.optimal_mask.dtype == np.dtype(np.bool_)
    assert tuple(result.policy_ids) == ("epsilon", "ucb")
    assert tuple(result.config) == (
        "horizon",
        "n_replicates",
        "seed",
        "reward_coupling",
        "record_contexts",
    )
    assert set(result.provenance.to_dict()) == {
        "pymab_version",
        "python_version",
        "numpy_version",
        "backend",
        "rng_scheme",
        "environment",
        "policies",
    }


def test_benchmark_cases_are_named_and_small_runs_build() -> None:
    assert CASE_NAMES == (
        "stationary",
        "bernoulli",
        "nonstationary",
        "contextual",
    )
    for name in CASE_NAMES:
        experiment = build_experiment(name, horizon=2, n_replicates=1)
        assert experiment.config.horizon == 2
        assert experiment.config.n_replicates == 1


def test_reference_worker_emits_stable_measurement_schema() -> None:
    measurement = measure_case(
        "stationary",
        horizon=2,
        n_replicates=1,
        repetitions=1,
    )

    assert measurement["schema_version"] == 1
    assert measurement["backend"] == "python-reference"
    assert measurement["case"] == "stationary"
    assert measurement["horizon"] == 2
    assert measurement["n_replicates"] == 1
    assert measurement["n_policies"] > 0
    assert measurement["decisions"] == 2 * measurement["n_policies"]
    assert measurement["result_bytes"] > 0
    assert measurement["median_elapsed_seconds"] >= 0
    assert measurement["decisions_per_second"] > 0
    assert len(measurement["elapsed_seconds"]) == 1
    assert set(measurement["environment"]) == {
        "implementation",
        "machine",
        "numpy",
        "platform",
        "processor",
        "pymab",
        "python",
    }
