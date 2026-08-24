from __future__ import annotations

import pytest

from pymab.policies import GreedyPolicy
from tests.parity.conftest import validate_policy_fixture


def _complete_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_kind": "greedy",
        "config": {"n_arms": 2},
        "updates": [{"action": 0, "reward": 1.0}],
        "checkpoints": [],
        "recommendation": 0,
        "reset_state": {},
        "expected_error": None,
    }


def test_policy_fixture_loader_accepts_the_complete_contract() -> None:
    assert validate_policy_fixture(_complete_fixture())["policy_kind"] == "greedy"


def test_python_action_value_state_uses_the_shared_fixture_shape() -> None:
    policy = GreedyPolicy(n_arms=2, initial_value=0.5)
    policy.update(action=1, reward=1.0)

    assert policy._parity_state() == {
        "step": 1,
        "total_reward": 1.0,
        "counts": [0.0, 1.0],
        "estimates": [0.5, 1.0],
    }


@pytest.mark.parametrize("field", sorted(_complete_fixture()))
def test_policy_fixture_loader_rejects_missing_fields(field: str) -> None:
    fixture = _complete_fixture()
    fixture.pop(field)
    with pytest.raises(ValueError, match="missing"):
        validate_policy_fixture(fixture)


def test_policy_fixture_loader_rejects_unknown_fields() -> None:
    fixture = _complete_fixture()
    fixture["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_policy_fixture(fixture)
