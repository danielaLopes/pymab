from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pymab import _native
from tests.parity.conftest import load_policy_fixture

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "policies"
FIXTURE_NAMES = sorted(
    path.name
    for path in FIXTURE_ROOT.glob("*.json")
    if path.name not in {"registry.json", "schema.json"}
)


def _assert_snapshot_matches(actual: object, expected: object) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_snapshot_matches(actual[key], value)
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence)
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        return
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        return
    assert actual == expected


@pytest.mark.parametrize(
    "fixture_name", FIXTURE_NAMES, ids=[Path(name).stem for name in FIXTURE_NAMES]
)
def test_native_handle_matches_shared_policy_fixture(fixture_name: str) -> None:
    if not _native.native_available():
        pytest.skip("native extension has not been built in this environment")
    extension = importlib.import_module("pymab._pymab")
    fixture = load_policy_fixture(FIXTURE_ROOT / fixture_name)
    kind = cast(str, fixture["policy_kind"])
    configuration = cast(Mapping[str, object], fixture["config"])
    policy = extension._NativePolicy.create(kind, json.dumps(configuration))

    completed = 0
    for update in cast(list[Mapping[str, object]], fixture["updates"]):
        repeat = update.get("repeat", 1)
        assert isinstance(repeat, int) and not isinstance(repeat, bool)
        for _ in range(repeat):
            context = update.get("context")
            policy.update(update["action"], update["reward"], context)
            completed += 1

    final = cast(list[Mapping[str, object]], fixture["checkpoints"])[-1]
    assert completed == final["after_update"]
    _assert_snapshot_matches(json.loads(policy.state_json()), final["state"])

    recommendation = fixture["recommendation"]
    if isinstance(recommendation, Mapping):
        actual = policy.recommend_action(recommendation["context"])
        expected_action = recommendation["action"]
    else:
        actual = policy.recommend_action()
        expected_action = recommendation
    assert actual == expected_action

    policy.reset()
    _assert_snapshot_matches(json.loads(policy.state_json()), fixture["reset_state"])
