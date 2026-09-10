"""Comparable state-memory accounting for native and reference policies."""

from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from pymab import _native
from pymab._reference.registry import create_reference_policy
from pymab.policies.policy import ContextualPolicy

Backend = Literal["python", "rust"]
FIXTURE_ROOT = Path(__file__).parents[1] / "tests" / "fixtures" / "policies"


def deep_size(value: object, *, _seen: set[int] | None = None) -> int:
    """Estimate Python-owned memory recursively without following cycles twice."""

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return total + int(value.nbytes)
    if isinstance(value, Mapping):
        return total + sum(
            deep_size(key, _seen=seen) + deep_size(item, _seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return total + sum(deep_size(item, _seen=seen) for item in value)
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        total += deep_size(namespace, _seen=seen)
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if hasattr(value, slot):
            total += deep_size(getattr(value, slot), _seen=seen)
    return total


def measure_policy_states(
    backend: Backend, *, fixture_root: Path = FIXTURE_ROOT
) -> dict[str, int]:
    """Replay the shared traces and measure all 27 resulting policy states."""

    if backend == "rust" and not _native.native_available():
        raise RuntimeError("the native extension is required for Rust measurements")
    measurements: dict[str, int] = {}
    paths = sorted(
        path
        for path in fixture_root.glob("*.json")
        if path.name not in {"registry.json", "schema.json"}
    )
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        kind = cast(str, fixture["policy_kind"])
        config = cast(dict[str, object], fixture["config"])
        if backend == "rust":
            policy = _native.create_policy(
                kind, json.dumps(config, sort_keys=True, separators=(",", ":"))
            )
        else:
            policy = create_reference_policy(kind, config)
        for update in cast(list[dict[str, Any]], fixture["updates"]):
            for _ in range(cast(int, update.get("repeat", 1))):
                action = cast(int, update["action"])
                reward = cast(float, update["reward"])
                context = update.get("context")
                if backend == "rust":
                    policy.update(action, reward, context)
                elif isinstance(policy, ContextualPolicy):
                    matrix = np.asarray(context, dtype=float).reshape(
                        policy.n_arms, policy.n_features
                    )
                    policy.update(action=action, reward=reward, context=matrix)
                else:
                    policy.update(action=action, reward=reward)
        measurements[kind] = (
            int(policy.estimated_state_bytes())
            if backend == "rust"
            else deep_size(policy)
        )
    return measurements


__all__ = ["Backend", "deep_size", "measure_policy_states"]
