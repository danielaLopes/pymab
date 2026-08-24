from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pymab.simulation as simulation_module
from pymab.distributions import BernoulliReward, GaussianReward, RewardModel
from pymab.environments import BanditEnvironment, EnvironmentDynamics, GradualDrift
from pymab.errors import CompatibilityError, SerializationError
from pymab.policies import RandomPolicy, UCBPolicy
from pymab.policies.policy import ContextualPolicy, Policy
from pymab.results import SimulationResult
from pymab.simulation import Experiment, ExperimentConfig
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


def test_experiment_rejects_noninteger_action() -> None:
    class NonIntegerPolicy(FixedPolicy):
        def select_action(self, *, rng):
            return 0.5

    with pytest.raises(ValueError, match="must be an integer"):
        Experiment(
            environment=BanditEnvironment(means=np.array([0.0, 1.0])),
            policies={"bad": NonIntegerPolicy(2)},
            config=ExperimentConfig(horizon=1, n_replicates=1, seed=1),
        ).run()


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (np.array([np.nan, 0.0]), "invalid reward sample"),
        (np.array([0.0]), "returned shape"),
    ],
)
def test_experiment_validates_reward_model_output(sample, message) -> None:
    class InvalidReward(RewardModel):
        domain = RewardDomain.REAL

        def sample(self, means, rng):
            return sample

    with pytest.raises(ValueError, match=message):
        Experiment(
            environment=BanditEnvironment(
                means=np.array([0.0, 1.0]), reward_model=InvalidReward()
            ),
            policies={"fixed": FixedPolicy(2)},
            config=ExperimentConfig(horizon=1, n_replicates=1, seed=1),
        ).run()


def test_experiment_validates_environment_mean_shape() -> None:
    class InvalidEnvironment:
        n_arms = 2
        reward_model = GaussianReward()
        reward_domain = RewardDomain.REAL
        contextual = False

        def clone(self):
            return self

        def advance(self, *, step, rng) -> None:
            return None

        def expected_rewards(self):
            return np.array([0.0])

    with pytest.raises(ValueError, match="one expected reward per arm"):
        Experiment(
            environment=InvalidEnvironment(),
            policies={"fixed": FixedPolicy(2)},
            config=ExperimentConfig(horizon=1, n_replicates=1, seed=1),
        ).run()


def test_contextual_policy_rejects_missing_runtime_context() -> None:
    class MissingContextEnvironment:
        n_arms = 1
        n_features = 1
        reward_model = GaussianReward()
        reward_domain = RewardDomain.REAL
        contextual = True

        def clone(self):
            return self

        def context(self, rng):
            return None

        def expected_rewards(self, context):
            return np.array([0.0])

    class FixedContextualPolicy(ContextualPolicy):
        def select_action(self, *, context, rng):
            return 0

        def update(self, *, action, reward, context) -> None:
            return None

        def reset(self) -> None:
            return None

        def recommend_action(self, *, context):
            return 0

    with pytest.raises(CompatibilityError, match="requires context"):
        Experiment(
            environment=MissingContextEnvironment(),
            policies={"contextual": FixedContextualPolicy(n_arms=1, n_features=1)},
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


def test_npz_save_does_not_build_full_json_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _experiment({"fixed": FixedPolicy(3)})

    def reject_full_payload(self) -> dict[str, object]:
        pytest.fail("NPZ persistence must not call SimulationResult.to_dict()")

    monkeypatch.setattr(SimulationResult, "to_dict", reject_full_payload)
    destination = result.save_npz(tmp_path / "result")
    assert SimulationResult.load_npz(destination).equals(result)


def test_npz_roundtrip_with_context_tensor(tmp_path: Path) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["contexts"] = np.zeros((4, 25, 3, 1)).tolist()
    result = SimulationResult.from_dict(payload)
    destination = result.save_npz(tmp_path / "contextual-result")
    restored = SimulationResult.load_npz(destination)
    assert restored.contexts is not None
    np.testing.assert_array_equal(restored.contexts, result.contexts)


def test_simulation_result_has_one_canonical_domain_module() -> None:
    assert SimulationResult.__module__ == "pymab.results"
    assert not hasattr(simulation_module, "SimulationResult")


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
    with pytest.raises(SerializationError, match="suffix"):
        SimulationResult.load_npz(tmp_path / "wrong.json")


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


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (np.array(3), "metadata must be JSON text"),
        (np.array("[]"), "metadata must be an object"),
    ],
)
def test_npz_rejects_invalid_metadata(
    tmp_path: Path, metadata: np.ndarray, message: str
) -> None:
    path = tmp_path / "invalid-metadata.npz"
    np.savez(path, metadata=metadata)
    with pytest.raises(SerializationError, match=message):
        SimulationResult.load_npz(path)


def test_npz_schema_errors_include_source_path(tmp_path: Path) -> None:
    path = tmp_path / "missing-arrays.npz"
    np.savez(path, metadata=np.array(json.dumps({"schema_version": 3})))
    with pytest.raises(SerializationError, match=str(path)):
        SimulationResult.load_npz(path)


def test_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"schema_version": NaN}', encoding="utf-8")
    with pytest.raises(SerializationError, match="nonfinite JSON number"):
        SimulationResult.load_json(path)


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


@pytest.mark.parametrize("schema", [None, True, "3", 1])
def test_result_rejects_missing_or_invalid_schema(schema) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    if schema is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = schema
    with pytest.raises(SerializationError, match="schema"):
        SimulationResult.from_dict(payload)


def test_schema_two_uses_unknown_version_for_invalid_legacy_version() -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["schema_version"] = 2
    payload["library_version"] = 12
    payload.pop("provenance")
    migrated = SimulationResult.from_dict(payload)
    assert migrated.library_version == "unknown"
    assert migrated.provenance.pymab_version == "unknown"


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("library_version", 2, "library_version"),
        ("policy_ids", ("fixed",), "policy_ids"),
        ("replicate_seeds", (1,), "replicate_seeds"),
        ("config", [], "config"),
        ("metadata", [], "metadata"),
        ("provenance", [], "provenance"),
    ],
)
def test_result_schema_rejects_invalid_field_types(field, value, message) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload[field] = value
    with pytest.raises(SerializationError, match=message):
        SimulationResult.from_dict(payload)


def test_result_schema_rejects_invalid_provenance_string() -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    provenance = dict(payload["provenance"])
    provenance["rng_scheme"] = ""
    payload["provenance"] = provenance
    with pytest.raises(SerializationError, match="provenance.rng_scheme"):
        SimulationResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rewards", [[[0.0]]], "equal shapes"),
        ("recommendations", [[[0]]], "recommendations must match"),
        ("arm_means", [[[0.1, 0.5, 0.9]]], "replicate and step"),
        ("optimal_mask", [[[True]]], "optimal_mask must match"),
        (
            "contexts",
            np.zeros((1, 1, 3, 1)).tolist(),
            "contexts must have shape",
        ),
        ("policy_ids", ["fixed", "other"], "policy dimension"),
    ],
)
def test_result_rejects_mismatched_dimensions(field, value, message) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        SimulationResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_ids", [""], "non-empty"),
        ("replicate_seeds", [True, 2, 3, 4], "only integers"),
        ("replicate_seeds", [-1, 2, 3, 4], "non-negative"),
    ],
)
def test_result_rejects_invalid_identifiers(field, value, message) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        SimulationResult.from_dict(payload)


def test_result_rejects_duplicate_policy_ids_at_matching_dimension() -> None:
    payload = _experiment({"first": FixedPolicy(3), "second": FixedPolicy(3)}).to_dict()
    payload["policy_ids"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="unique"):
        SimulationResult.from_dict(payload)


def test_result_rejects_steps_without_an_optimal_arm() -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload["optimal_mask"] = np.zeros((4, 25, 3), dtype=bool).tolist()
    with pytest.raises(ValueError, match="at least one optimal arm"):
        SimulationResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actions", 99, "actions contain"),
        ("recommendations", 99, "recommendations contain"),
        ("expected_rewards", 99.0, "selected arm means"),
    ],
)
def test_result_rejects_invalid_observation_values(field, value, message) -> None:
    payload = _experiment({"fixed": FixedPolicy(3)}).to_dict()
    payload[field][0][0][0] = value
    with pytest.raises(ValueError, match=message):
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
