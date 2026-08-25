"""Versioned payload schema for simulation results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np

from pymab.errors import SerializationError
from pymab.provenance import JSONValue, RunProvenance, thaw_json
from pymab.types import BoolArray, FloatArray, IntArray

if TYPE_CHECKING:
    from pymab.results import SimulationResult

SCHEMA_VERSION = 3
ARRAY_FIELDS = (
    "rewards",
    "actions",
    "expected_rewards",
    "arm_means",
    "optimal_mask",
    "recommendations",
)


def _metadata_payload(result: SimulationResult) -> dict[str, object]:
    """Build JSON metadata without traversing result arrays."""

    return {
        "schema_version": result.schema_version,
        "library_version": result.library_version,
        "policy_ids": list(result.policy_ids),
        "replicate_seeds": list(result.replicate_seeds),
        "config": thaw_json(result.config),
        "metadata": thaw_json(result.metadata),
        "provenance": result.provenance.to_dict(),
        "context_digest": result.context_digest,
    }


def _complete_payload(result: SimulationResult) -> dict[str, object]:
    """Build a complete JSON-compatible result payload."""

    payload = _metadata_payload(result)
    payload.update(
        {
            "contexts": (None if result.contexts is None else result.contexts.tolist()),
            "rewards": result.rewards.tolist(),
            "actions": result.actions.tolist(),
            "expected_rewards": result.expected_rewards.tolist(),
            "arm_means": result.arm_means.tolist(),
            "optimal_mask": result.optimal_mask.tolist(),
            "recommendations": result.recommendations.tolist(),
        }
    )
    return payload


def _result_from_payload(payload: Mapping[str, object]) -> SimulationResult:
    """Migrate, validate, and construct a result from an external payload."""

    migrated = _migrate_payload(payload)
    _validate_fields(migrated)
    provenance_payload = cast(Mapping[str, object], migrated["provenance"])
    provenance = RunProvenance(
        pymab_version=_required_string(provenance_payload, "pymab_version"),
        python_version=_required_string(provenance_payload, "python_version"),
        numpy_version=_required_string(provenance_payload, "numpy_version"),
        rng_scheme=_legacy_compatible_string(provenance_payload, "rng_scheme"),
        environment=cast(Mapping[str, JSONValue], provenance_payload["environment"]),
        policies=cast(Mapping[str, JSONValue], provenance_payload["policies"]),
        backend=_backend_string(provenance_payload),
    )

    from pymab.results import SimulationResult

    return SimulationResult(
        rewards=cast(FloatArray, migrated["rewards"]),
        actions=cast(IntArray, migrated["actions"]),
        expected_rewards=cast(FloatArray, migrated["expected_rewards"]),
        arm_means=cast(FloatArray, migrated["arm_means"]),
        optimal_mask=cast(BoolArray, migrated["optimal_mask"]),
        recommendations=cast(IntArray, migrated["recommendations"]),
        policy_ids=tuple(cast(list[str], migrated["policy_ids"])),
        replicate_seeds=tuple(cast(list[int], migrated["replicate_seeds"])),
        config=cast(Mapping[str, JSONValue], migrated["config"]),
        metadata=cast(Mapping[str, JSONValue], migrated["metadata"]),
        provenance=provenance,
        contexts=cast(FloatArray | None, migrated.get("contexts")),
        context_digest=cast(str | None, migrated.get("context_digest")),
        schema_version=cast(int, migrated["schema_version"]),
        library_version=cast(str, migrated["library_version"]),
    )


def _migrate_payload(payload: Mapping[str, object]) -> dict[str, object]:
    try:
        schema = payload["schema_version"]
    except KeyError as exc:
        raise SerializationError("result payload is missing schema_version") from exc
    if isinstance(schema, bool) or not isinstance(schema, (int, np.integer)):
        raise SerializationError("schema_version must be an integer")
    schema_value = int(schema)
    if schema_value == SCHEMA_VERSION:
        return dict(payload)
    if schema_value == 2:
        library_version = payload.get("library_version", "unknown")
        if not isinstance(library_version, str):
            library_version = "unknown"
        migrated = dict(payload)
        migrated["schema_version"] = SCHEMA_VERSION
        migrated["library_version"] = library_version
        migrated["contexts"] = None
        migrated["context_digest"] = None
        migrated["provenance"] = RunProvenance.unknown(
            pymab_version=library_version
        ).to_dict()
        return migrated
    raise SerializationError(
        f"unsupported result schema {schema_value}; expected {SCHEMA_VERSION} or 2"
    )


def _validate_fields(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "library_version",
        "policy_ids",
        "replicate_seeds",
        "config",
        "metadata",
        "provenance",
        *ARRAY_FIELDS,
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise SerializationError(f"result payload is missing fields: {missing}")
    if not isinstance(payload["library_version"], str):
        raise SerializationError("library_version must be a string")
    if not isinstance(payload["policy_ids"], list):
        raise SerializationError("policy_ids must be a list")
    if not isinstance(payload["replicate_seeds"], list):
        raise SerializationError("replicate_seeds must be a list")
    for name in ("config", "metadata", "provenance"):
        if not isinstance(payload[name], Mapping):
            raise SerializationError(f"{name} must be an object")


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise SerializationError(f"provenance.{field} must be a non-empty string")
    return value


def _legacy_compatible_string(payload: Mapping[str, object], field: str) -> str:
    """Read a provenance field added after result schema three was introduced."""

    if field not in payload:
        return "unknown"
    return _required_string(payload, field)


def _backend_string(payload: Mapping[str, object]) -> str:
    backend = _legacy_compatible_string(payload, "backend")
    if backend not in {"python", "rust", "unknown"}:
        raise SerializationError(
            "provenance.backend must be 'python', 'rust', or 'unknown'"
        )
    return backend


__all__: list[str] = []
