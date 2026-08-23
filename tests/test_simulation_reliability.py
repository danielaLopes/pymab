from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pymab.distributions import BernoulliReward, GaussianReward, RewardModel
from pymab.environments import BanditEnvironment, EnvironmentDynamics, GradualDrift
from pymab.errors import SerializationError
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.policies.policy import Policy
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult
from pymab.types import RewardDomain


class FixedPolicy(Policy):
    def __init__(self, n_arms: int, action: int = 0) -> None:
        self.action = action
        super().__init__(n_arms=n_arms)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return self.action

    def update(self, *, action: int, reward: float) -> None:
        return None

    def reset(self) -> None:
        return None

    def recommend_action(self) -> int:
        return self.action


def _experiment(policies, *, coupling="common") -> SimulationResult:
    return Experiment(
        environment=BanditEnvironment(
            means=np.array([0.1, 0.5, 0.9]),
            reward_model=GaussianReward(std=0.2),
            dynamics=GradualDrift(std=0.02),
        ),
        policies=policies,
        config=ExperimentConfig(
            horizon=25,
            n_replicates=4,
            seed=123,
            reward_coupling=coupling,
        ),
        metadata={"scenario": "reliability-test", "tags": ["common-rng"]},
    ).run()


def test_seed_is_reproducible() -> None:
    left = _experiment({"ucb": UCBPolicy(n_arms=3)})
    right = _experiment({"ucb": UCBPolicy(n_arms=3)})
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.arm_means, right.arm_means)
    assert left.replicate_seeds == right.replicate_seeds


def test_adding_or_reordering_policies_does_not_change_existing_trace() -> None:
    alone = _experiment({"ucb": UCBPolicy(n_arms=3)})
    before = _experiment({"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)})
    after = _experiment({"ucb": UCBPolicy(n_arms=3), "random": RandomPolicy(n_arms=3)})
    for candidate, index in [(before, 1), (after, 0)]:
        np.testing.assert_array_equal(
            alone.actions[:, :, 0], candidate.actions[:, :, index]
        )
        np.testing.assert_array_equal(
            alone.rewards[:, :, 0], candidate.rewards[:, :, index]
        )
        np.testing.assert_array_equal(alone.arm_means, candidate.arm_means)


def test_reward_coupling_modes() -> None:
    common = _experiment({"a": FixedPolicy(3), "b": FixedPolicy(3)}, coupling="common")
    independent = _experiment(
        {"a": FixedPolicy(3), "b": FixedPolicy(3)}, coupling="independent"
    )
    np.testing.assert_array_equal(common.rewards[:, :, 0], common.rewards[:, :, 1])
    assert not np.array_equal(
        independent.rewards[:, :, 0], independent.rewards[:, :, 1]
    )


def test_mutable_environment_components_are_isolated_per_replicate() -> None:
    class StatefulReward(RewardModel):
        domain = RewardDomain.REAL
        clones: list[StatefulReward] = []

        def __init__(self) -> None:
            self.calls = 0

        def sample(self, means, rng):
            self.calls += 1
            return np.asarray(means, dtype=float)

        def clone(self):
            clone = type(self)()
            type(self).clones.append(clone)
            return clone

    class StatefulDynamics(EnvironmentDynamics):
        supported_domains = frozenset({RewardDomain.REAL})
        clones: list[StatefulDynamics] = []

        def __init__(self) -> None:
            self.calls = 0

        def apply(self, means, *, step, rng):
            self.calls += 1
            return np.asarray(means, dtype=float)

        def clone(self):
            clone = type(self)()
            type(self).clones.append(clone)
            return clone

    reward_model = StatefulReward()
    dynamics = StatefulDynamics()
    Experiment(
        environment=BanditEnvironment(
            means=np.array([0.0]),
            reward_model=reward_model,
            dynamics=dynamics,
        ),
        policies={"fixed": FixedPolicy(1)},
        config=ExperimentConfig(horizon=3, n_replicates=2, seed=4),
    ).run()
    assert reward_model.calls == dynamics.calls == 0
    assert [clone.calls for clone in StatefulReward.clones] == [3, 3]
    assert [clone.calls for clone in StatefulDynamics.clones] == [3, 3]


def test_tied_arms_all_count_as_optimal() -> None:
    result = Experiment(
        environment=BanditEnvironment(
            means=np.array([1.0, 1.0]), reward_model=GaussianReward(std=0.1)
        ),
        policies={"right": FixedPolicy(2, action=1)},
        config=ExperimentConfig(horizon=3, n_replicates=2, seed=2),
    ).run()
    assert np.all(result.optimal_mask)
    np.testing.assert_array_equal(result.regret, 0)
    np.testing.assert_array_equal(result.simple_regret, 0)
    np.testing.assert_array_equal(result.optimal_action_rate_by_step, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0, "n_replicates": 1, "seed": 1},
        {"horizon": 1.5, "n_replicates": 1, "seed": 1},
        {"horizon": 1, "n_replicates": 0, "seed": 1},
        {"horizon": 1, "n_replicates": 1, "seed": None},
        {
            "horizon": 1,
            "n_replicates": 1,
            "seed": 1,
            "reward_coupling": "invalid",
        },
    ],
)
def test_config_validation(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(**kwargs)


def test_experiment_validates_policy_contracts_before_run() -> None:
    environment = BanditEnvironment(
        means=np.array([0.2, 0.8]), reward_model=BernoulliReward()
    )
    config = ExperimentConfig(horizon=2, n_replicates=1, seed=1)
    with pytest.raises(ValueError, match="at least one"):
        Experiment(environment=environment, policies={}, config=config)
    with pytest.raises(ValueError, match="non-empty"):
        Experiment(
            environment=environment,
            policies={" ": RandomPolicy(n_arms=2)},
            config=config,
        )
    with pytest.raises(TypeError, match="n_arms"):
        Experiment(
            environment=environment,
            policies={"random": RandomPolicy(n_arms=3)},
            config=config,
        )


def test_experiment_rejects_invalid_action() -> None:
    with pytest.raises(ValueError, match="outside"):
        Experiment(
            environment=BanditEnvironment(means=np.array([0.0, 1.0])),
            policies={"bad": FixedPolicy(2, action=5)},
            config=ExperimentConfig(horizon=1, n_replicates=1, seed=1),
        ).run()


def test_result_arrays_are_read_only_and_metadata_is_immutable() -> None:
    result = _experiment({"fixed": FixedPolicy(3)})
    with pytest.raises(ValueError):
        result.rewards[0, 0, 0] = 99
    with pytest.raises(TypeError):
        result.config["seed"] = 99
    with pytest.raises(TypeError):
        result.metadata["scenario"] = "changed"
    assert result.metadata["tags"] == ("common-rng",)


def test_result_npz_and_json_roundtrip() -> None:
    result = _experiment({"fixed": FixedPolicy(3)})
    with tempfile.TemporaryDirectory() as directory:
        npz_path = Path(directory) / "result.npz"
        json_path = Path(directory) / "result.json"
        result.save_npz(npz_path)
        result.save_json(json_path)
        from_npz = SimulationResult.load_npz(npz_path)
        from_json = SimulationResult.load_json(json_path)
    for loaded in (from_npz, from_json):
        np.testing.assert_array_equal(loaded.rewards, result.rewards)
        np.testing.assert_array_equal(loaded.optimal_mask, result.optimal_mask)
        assert loaded.policy_ids == result.policy_ids
        assert loaded.replicate_seeds == result.replicate_seeds
        assert dict(loaded.config) == dict(result.config)
        assert dict(loaded.metadata) == dict(result.metadata)
        assert loaded.equals(result)
        assert loaded.provenance.equals(result.provenance)


def test_result_paths_are_suffix_consistent(tmp_path: Path) -> None:
    result = _experiment({"fixed": FixedPolicy(3)})
    npz_path = result.save_npz(tmp_path / "archive")
    json_path = result.save_json(tmp_path / "payload")
    assert npz_path.suffix == ".npz"
    assert json_path.suffix == ".json"
    assert SimulationResult.load_npz(tmp_path / "archive").equals(result)
    assert SimulationResult.load_json(tmp_path / "payload").equals(result)
    with pytest.raises(SerializationError, match="suffix"):
        result.save_npz(tmp_path / "wrong.json")


def test_atomic_write_preserves_destination_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _experiment({"fixed": FixedPolicy(3)})
    destination = tmp_path / "result.json"
    destination.write_text("original", encoding="utf-8")

    def fail_replace(source, target) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("pymab.persistence.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        result.save_json(destination)
    assert destination.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


@pytest.mark.parametrize("suffix", ["json", "npz"])
def test_corrupt_persistence_is_wrapped(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"corrupt.{suffix}"
    path.write_bytes(b"not a valid result")
    loader = (
        SimulationResult.load_json if suffix == "json" else SimulationResult.load_npz
    )
    with pytest.raises(SerializationError, match="could not load"):
        loader(path)


def test_schema_two_payload_has_explicit_unknown_provenance() -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["schema_version"] = 2
    payload.pop("provenance")
    payload.pop("contexts")
    payload.pop("context_digest")
    migrated = SimulationResult.from_dict(payload)
    assert migrated.schema_version == 3
    assert migrated.provenance.python_version == "unknown"


def test_result_rejects_unknown_schema() -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["schema_version"] = 999
    with pytest.raises(SerializationError, match="unsupported"):
        SimulationResult.from_dict(payload)


@pytest.mark.parametrize(
    "metadata", [{1: "value"}, {"value": np.nan}, {"value": object()}]
)
def test_result_rejects_non_json_metadata(metadata) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["metadata"] = metadata
    with pytest.raises((TypeError, ValueError)):
        SimulationResult.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(policy_ids=[]),
        lambda payload: payload.update(policy_ids=["fixed", "fixed"]),
        lambda payload: payload.update(replicate_seeds=[]),
        lambda payload: payload.update(recommendations=[[[99]]]),
        lambda payload: payload.update(rewards=[[[np.nan]]]),
        lambda payload: payload.update(optimal_mask=[[[False, False, False]]]),
    ],
)
def test_result_validation_rejects_inconsistent_data(mutation) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    mutation(payload)
    with pytest.raises(ValueError):
        SimulationResult.from_dict(payload)


def test_result_json_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([]))
    with pytest.raises(SerializationError, match="object"):
        SimulationResult.load_json(path)


def test_result_json_schema_errors_include_source_path(tmp_path: Path) -> None:
    path = tmp_path / "missing-fields.json"
    path.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    with pytest.raises(SerializationError, match=str(path)):
        SimulationResult.load_json(path)


@pytest.mark.optional
def test_result_to_pandas_is_tidy() -> None:
    result = _experiment({"fixed": FixedPolicy(3)})
    frame = result.to_pandas()
    assert len(frame) == result.n_replicates * result.horizon * result.n_policies
    assert {"policy_id", "replicate_seed", "simple_regret"} <= set(frame.columns)
