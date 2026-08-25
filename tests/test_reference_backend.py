from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

import pymab.policies as public_policies
from pymab._reference.registry import (
    REFERENCE_POLICY_SPECS,
    clone_reference_policy,
    create_reference_policy,
    reference_policy_config,
    reference_policy_kind,
)
from pymab.policies.policy import ActionValuePolicy

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "policies"


def _registry_entries() -> list[Mapping[str, str]]:
    payload = json.loads((FIXTURE_ROOT / "registry.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    entries = payload["policies"]
    assert isinstance(entries, list)
    return entries


def _fixture_configs() -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for path in FIXTURE_ROOT.glob("*.json"):
        if path.name in {"registry.json", "schema.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        kind = payload["policy_kind"]
        config = payload["config"]
        assert isinstance(kind, str)
        assert isinstance(config, dict)
        result[kind] = config
    return result


def test_reference_registry_constructs_every_public_builtin() -> None:
    entries = _registry_entries()
    configs = _fixture_configs()
    assert set(REFERENCE_POLICY_SPECS) == {entry["rust_kind"] for entry in entries}
    assert set(configs) == set(REFERENCE_POLICY_SPECS)

    for entry in entries:
        kind = entry["rust_kind"]
        class_name = entry["python_name"]
        policy = create_reference_policy(kind, configs[kind])
        assert type(policy).__name__ == class_name
        assert type(policy).__module__.startswith("pymab._reference.policies.")
        assert getattr(public_policies, class_name) is type(policy)
        assert reference_policy_kind(policy) == kind
        assert dict(reference_policy_config(policy)) == dict(configs[kind])


def test_reference_clone_uses_immutable_config_and_fresh_state() -> None:
    policy = public_policies.UCBPolicy(
        n_arms=2,
        initial_value=0.25,
        c=1.5,
        reward_scale=2.0,
    )
    policy.update(action=0, reward=1.0)
    clone = clone_reference_policy(policy)

    assert isinstance(clone, public_policies.UCBPolicy)
    assert clone is not policy
    assert clone.step == 0
    np.testing.assert_array_equal(clone.counts, [0.0, 0.0])
    config = reference_policy_config(policy)
    with pytest.raises(TypeError):
        config["n_arms"] = 3  # type: ignore[index]


def test_reference_registry_rejects_unknown_or_incomplete_configuration() -> None:
    with pytest.raises(ValueError, match="unknown reference policy"):
        create_reference_policy("not-a-policy", {})
    with pytest.raises(ValueError, match="configuration fields differ"):
        create_reference_policy("greedy", {"n_arms": 2})
    with pytest.raises(ValueError, match="configuration fields differ"):
        create_reference_policy(
            "random", {"n_arms": 2, "surprise": True}
        )


def test_custom_action_value_subclasses_remain_public_and_unregistered() -> None:
    class CustomPolicy(ActionValuePolicy):
        def select_action(self, *, rng: np.random.Generator) -> int:
            del rng
            return 0

    policy = CustomPolicy(n_arms=2)
    policy.update(action=0, reward=1.0)
    assert policy.recommend_action() == 0
    with pytest.raises(TypeError, match="custom policies"):
        reference_policy_kind(policy)
