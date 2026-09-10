"""Shared implementation for thin public wrappers around native policy state."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType
from typing import Any, TypeVar, cast

import numpy as np

from pymab import _native
from pymab._reference.registry import (
    ReferencePolicy,
    reference_policy_config,
)
from pymab.types import FloatArray

_ReferenceType = TypeVar("_ReferenceType", bound=type[Any])

_ARRAY_FIELDS = frozenset(
    {
        "a",
        "active",
        "b",
        "change_counts",
        "counts",
        "detector_counts",
        "detector_means",
        "discounted_counts",
        "discounted_sums",
        "estimates",
        "failures",
        "last_probabilities",
        "log_weights",
        "means",
        "negative_cusum",
        "ph_cumulative",
        "ph_minimum",
        "phase_counts",
        "phase_sums",
        "positive_cusum",
        "precisions",
        "preferences",
        "probabilities",
        "successes",
        "theta",
    }
)
_SCALAR_FIELDS = frozenset(
    {
        "average_reward",
        "history_len",
        "phase_delta",
        "phase_epsilon",
        "step",
        "total_reward",
    }
)


class NativePolicyMixin:
    """Route the public policy protocol to one opaque Rust-owned handle."""

    _native_handle: Any
    _native_kind: str
    _native_configuration: Mapping[str, object]
    n_arms: int
    n_features: int

    def _initialize_native(
        self,
        *,
        kind: str,
        configuration: Mapping[str, object],
        handle: Any | None = None,
    ) -> None:
        normalized = dict(configuration)
        self._native_kind = kind
        self._native_configuration = MappingProxyType(normalized)
        for field, value in normalized.items():
            setattr(self, field, value)
        self._native_handle = handle or _native.create_policy(
            kind, json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        )

    @property
    def backend(self) -> str:
        """Return the policy-state backend in use."""

        return "rust"

    @property
    def configuration(self) -> Mapping[str, object]:
        """Return immutable normalized constructor configuration."""

        return self._native_configuration

    def _state_snapshot(self) -> dict[str, object]:
        payload = json.loads(cast(str, self._native_handle.state_json()))
        if not isinstance(payload, dict):
            raise RuntimeError("native policy returned a non-object state snapshot")
        return cast(dict[str, object], payload)

    @property
    def step(self) -> int:
        """Return the number of completed native updates."""

        value = self._state_snapshot().get("step")
        if not isinstance(value, int):
            raise AttributeError("step")
        return value

    @property
    def total_reward(self) -> float:
        """Return cumulative reward observed by the native policy."""

        value = self._state_snapshot().get("total_reward")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise AttributeError("total_reward")
        return float(value)

    def __getattr__(self, name: str) -> object:
        if name == "_history":
            length = self._state_snapshot().get("history_len")
            if isinstance(length, int):
                return (None,) * length
        if name not in _ARRAY_FIELDS and name not in _SCALAR_FIELDS:
            raise AttributeError(name)
        state = self._state_snapshot()
        if name not in state:
            raise AttributeError(name)
        value = state[name]
        if name in _SCALAR_FIELDS:
            return value
        dtype: type[np.bool_] | type[np.float64]
        dtype = np.bool_ if name == "active" else np.float64
        result = np.asarray(value, dtype=dtype)
        if name == "a":
            result = result.reshape(self.n_arms, self.n_features, self.n_features)
        elif name in {"b", "theta"}:
            result = result.reshape(self.n_arms, self.n_features)
        result.setflags(write=False)
        return result

    def select_action(
        self,
        *,
        rng: np.random.Generator,
        context: FloatArray | None = None,
    ) -> int:
        if context is not None:
            self._validate_context(context)
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        flat_context = None if context is None else context.reshape(-1).tolist()
        return int(self._native_handle.select_action(seed, flat_context))

    def _validate_action(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= int(action) < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")

    def _validate_context(self, context: FloatArray) -> None:
        if context.shape != (self.n_arms, self.n_features):
            raise ValueError("context must have shape (n_arms, n_features)")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")

    def update(
        self,
        *,
        action: int,
        reward: float,
        context: FloatArray | None = None,
    ) -> None:
        self._validate_action(action)
        if context is not None:
            self._validate_context(context)
        flat_context = None if context is None else context.reshape(-1).tolist()
        self._native_handle.update(action, float(reward), flat_context)

    def recommend_action(self, *, context: FloatArray | None = None) -> int:
        if context is not None:
            self._validate_context(context)
        flat_context = None if context is None else context.reshape(-1).tolist()
        return int(self._native_handle.recommend_action(flat_context))

    def reset(self) -> None:
        self._native_handle.reset()

    def clone(self) -> ReferencePolicy:
        clone = object.__new__(type(self))
        clone._initialize_native(
            kind=self._native_kind,
            configuration=self._native_configuration,
            handle=self._native_handle.clone_reset(),
        )
        return cast(ReferencePolicy, clone)

    def estimated_state_bytes(self) -> int:
        """Estimate Rust-owned state memory including reserved capacities."""

        return int(self._native_handle.estimated_state_bytes())

    def _parity_state(self) -> dict[str, object]:
        state = self._state_snapshot()
        return {
            field: state[field]
            for field in ("step", "total_reward", "counts", "estimates")
            if field in state
        }


def native_policy_class(
    kind: str, reference_type: _ReferenceType, *, module: str
) -> _ReferenceType:
    """Build a runtime wrapper class or return the reference type as fallback."""

    if not _native.native_available():
        return reference_type

    def __init__(self: NativePolicyMixin, *args: object, **kwargs: object) -> None:
        reference = cast(Any, reference_type)(*args, **kwargs)
        self._initialize_native(
            kind=kind,
            configuration=reference_policy_config(reference),
        )

    wrapper = type(
        reference_type.__name__,
        (NativePolicyMixin, reference_type),
        {
            "__doc__": reference_type.__doc__,
            "__init__": __init__,
            "__module__": module,
            "__signature__": inspect.signature(reference_type),
        },
    )
    return cast(_ReferenceType, wrapper)


__all__ = ["NativePolicyMixin", "native_policy_class"]
