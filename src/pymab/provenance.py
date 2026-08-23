"""Immutable, JSON-compatible experiment provenance."""

from __future__ import annotations

import inspect
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np

from pymab._random import RNG_SCHEME_VERSION
from pymab.errors import ValidationError

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]


def freeze_json(value: object, *, name: str = "value") -> JSONValue:
    """Return a recursively immutable JSON value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError(f"{name} numbers must be finite")
        return value
    if isinstance(value, np.generic):
        return freeze_json(value.item(), name=name)
    if isinstance(value, Enum):
        return freeze_json(value.value, name=name)
    if isinstance(value, np.ndarray):
        return tuple(freeze_json(item, name=name) for item in value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError(f"{name} keys must be strings")
        return MappingProxyType(
            {
                str(key): freeze_json(item, name=f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, name=name) for item in value)
    raise ValidationError(f"{name} must contain only JSON-compatible values")


def thaw_json(value: JSONValue) -> JSONScalar | list[object] | dict[str, object]:
    """Convert an immutable JSON value into standard JSON containers."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def component_snapshot(component: object) -> Mapping[str, JSONValue]:
    """Describe a configured component without serializing learned state."""

    parameters: dict[str, JSONValue] = {}
    seen: set[str] = set()
    for component_type in reversed(type(component).mro()):
        initializer = component_type.__dict__.get("__init__")
        if initializer is None:
            continue
        try:
            signature = inspect.signature(initializer)
        except (TypeError, ValueError):
            continue
        for parameter in signature.parameters.values():
            if parameter.name == "self" or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.name in seen or not hasattr(component, parameter.name):
                continue
            seen.add(parameter.name)
            value = getattr(component, parameter.name)
            parameters[parameter.name] = _snapshot_value(
                value, name=f"{type(component).__name__}.{parameter.name}"
            )
    return cast(
        Mapping[str, JSONValue],
        freeze_json(
            {
                "class": f"{type(component).__module__}.{type(component).__qualname__}",
                "parameters": parameters,
            },
            name="component",
        ),
    )


def _snapshot_value(value: object, *, name: str) -> JSONValue:
    if callable(value) and not isinstance(value, type):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"
    if _is_component(value):
        return component_snapshot(value)
    return freeze_json(value, name=name)


def _is_component(value: object) -> bool:
    return hasattr(value, "__dict__") and not isinstance(
        value, (str, bytes, int, float, bool, tuple, list, Mapping, np.ndarray)
    )


@dataclass(frozen=True, eq=False)
class RunProvenance:
    """Runtime and component configuration needed to audit a simulation."""

    pymab_version: str
    python_version: str
    numpy_version: str
    rng_scheme: str
    environment: Mapping[str, JSONValue]
    policies: Mapping[str, JSONValue]

    @classmethod
    def unknown(cls, *, pymab_version: str) -> RunProvenance:
        """Create an explicit placeholder when migrating legacy results."""

        return cls(
            pymab_version=pymab_version,
            python_version="unknown",
            numpy_version="unknown",
            rng_scheme="unknown",
            environment=cast(
                Mapping[str, JSONValue], freeze_json({"class": "unknown"})
            ),
            policies=cast(Mapping[str, JSONValue], freeze_json({})),
        )

    @classmethod
    def capture(
        cls,
        *,
        pymab_version: str,
        environment: object,
        policies: Mapping[str, object],
    ) -> RunProvenance:
        """Capture immutable runtime and component configuration."""

        return cls(
            pymab_version=pymab_version,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            rng_scheme=RNG_SCHEME_VERSION,
            environment=component_snapshot(environment),
            policies=cast(
                Mapping[str, JSONValue],
                freeze_json(
                    {
                        name: component_snapshot(policy)
                        for name, policy in policies.items()
                    },
                    name="policies",
                ),
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            cast(
                Mapping[str, JSONValue],
                freeze_json(self.environment, name="environment"),
            ),
        )
        object.__setattr__(
            self,
            "policies",
            cast(Mapping[str, JSONValue], freeze_json(self.policies, name="policies")),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable provenance payload."""

        return {
            "pymab_version": self.pymab_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "rng_scheme": self.rng_scheme,
            "environment": thaw_json(self.environment),
            "policies": thaw_json(self.policies),
        }

    def equals(self, other: object) -> bool:
        """Return whether another provenance record has the same value."""

        return isinstance(other, RunProvenance) and self.to_dict() == other.to_dict()


__all__ = [
    "JSONScalar",
    "JSONValue",
    "RunProvenance",
    "component_snapshot",
    "freeze_json",
    "thaw_json",
]
