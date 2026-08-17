"""Atomic, schema-aware persistence for simulation results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, cast

import numpy as np

from pymab.errors import SerializationError, ValidationError
from pymab.provenance import JSONValue, RunProvenance
from pymab.results import SCHEMA_VERSION, SimulationResult
from pymab.types import BoolArray, FloatArray, IntArray


class ResultSerializer:
    """Serialize and migrate :class:`pymab.results.SimulationResult` payloads."""

    @classmethod
    def save_json(cls, result: SimulationResult, path: str | Path) -> Path:
        destination = cls._normalize_path(path, suffix=".json")

        def write(handle: IO[bytes]) -> None:
            data = json.dumps(
                result.to_dict(), indent=2, allow_nan=False, ensure_ascii=False
            ).encode("utf-8")
            handle.write(data)

        cls._atomic_write(destination, write)
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> SimulationResult:
        source = cls._resolve_source(path, suffix=".json")
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=lambda value: _raise_nonfinite_json(value),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise SerializationError(
                f"could not load result JSON {source}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SerializationError(f"result JSON {source} must contain an object")
        try:
            return cls.from_payload(cast(Mapping[str, object], payload))
        except (KeyError, TypeError, ValueError, SerializationError) as exc:
            raise SerializationError(f"invalid result JSON {source}: {exc}") from exc

    @classmethod
    def save_npz(cls, result: SimulationResult, path: str | Path) -> Path:
        destination = cls._normalize_path(path, suffix=".npz")
        metadata = result.to_dict()
        for key in (
            "rewards",
            "actions",
            "expected_rewards",
            "arm_means",
            "optimal_mask",
            "recommendations",
            "contexts",
        ):
            metadata.pop(key)

        def write(handle: IO[bytes]) -> None:
            if result.contexts is not None:
                np.savez_compressed(
                    handle,
                    rewards=result.rewards,
                    actions=result.actions,
                    expected_rewards=result.expected_rewards,
                    arm_means=result.arm_means,
                    optimal_mask=result.optimal_mask,
                    recommendations=result.recommendations,
                    contexts=result.contexts,
                    metadata=np.array(json.dumps(metadata, allow_nan=False)),
                )
            else:
                np.savez_compressed(
                    handle,
                    rewards=result.rewards,
                    actions=result.actions,
                    expected_rewards=result.expected_rewards,
                    arm_means=result.arm_means,
                    optimal_mask=result.optimal_mask,
                    recommendations=result.recommendations,
                    metadata=np.array(json.dumps(metadata, allow_nan=False)),
                )

        cls._atomic_write(destination, write)
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> SimulationResult:
        source = cls._resolve_source(path, suffix=".npz")
        try:
            with np.load(source, allow_pickle=False) as archive:
                metadata_raw = archive["metadata"].item()
                if not isinstance(metadata_raw, str):
                    raise SerializationError("archive metadata must be JSON text")
                metadata = json.loads(
                    metadata_raw,
                    parse_constant=lambda value: _raise_nonfinite_json(value),
                )
                if not isinstance(metadata, dict):
                    raise SerializationError("archive metadata must be an object")
                payload: dict[str, object] = dict(metadata)
                for key in (
                    "rewards",
                    "actions",
                    "expected_rewards",
                    "arm_means",
                    "optimal_mask",
                    "recommendations",
                ):
                    payload[key] = archive[key].copy()
                payload["contexts"] = (
                    archive["contexts"].copy() if "contexts" in archive.files else None
                )
        except SerializationError as exc:
            raise SerializationError(
                f"could not load result archive {source}: {exc}"
            ) from exc
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SerializationError(
                f"could not load result archive {source}: {exc}"
            ) from exc
        try:
            return cls.from_payload(payload)
        except (KeyError, TypeError, ValueError, SerializationError) as exc:
            raise SerializationError(f"invalid result archive {source}: {exc}") from exc

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SimulationResult:
        migrated = cls._migrate(payload)
        cls._validate_fields(migrated)
        provenance_payload = cast(Mapping[str, object], migrated["provenance"])
        provenance = RunProvenance(
            pymab_version=_required_string(provenance_payload, "pymab_version"),
            python_version=_required_string(provenance_payload, "python_version"),
            numpy_version=_required_string(provenance_payload, "numpy_version"),
            rng_scheme=_required_string(provenance_payload, "rng_scheme"),
            environment=cast(
                Mapping[str, JSONValue], provenance_payload["environment"]
            ),
            policies=cast(Mapping[str, JSONValue], provenance_payload["policies"]),
        )
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

    @classmethod
    def _migrate(cls, payload: Mapping[str, object]) -> dict[str, object]:
        try:
            schema = payload["schema_version"]
        except KeyError as exc:
            raise SerializationError(
                "result payload is missing schema_version"
            ) from exc
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
            migrated["contexts"] = None
            migrated["context_digest"] = None
            migrated["provenance"] = RunProvenance.unknown(
                pymab_version=library_version
            ).to_dict()
            return migrated
        raise SerializationError(
            f"unsupported result schema {schema_value}; expected {SCHEMA_VERSION} or 2"
        )

    @staticmethod
    def _validate_fields(payload: Mapping[str, object]) -> None:
        required = {
            "schema_version",
            "library_version",
            "policy_ids",
            "replicate_seeds",
            "config",
            "metadata",
            "provenance",
            "rewards",
            "actions",
            "expected_rewards",
            "arm_means",
            "optimal_mask",
            "recommendations",
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

    @staticmethod
    def _normalize_path(path: str | Path, *, suffix: str) -> Path:
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(suffix)
        elif destination.suffix.lower() != suffix:
            raise SerializationError(f"path must use the {suffix} suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    @classmethod
    def _resolve_source(cls, path: str | Path, *, suffix: str) -> Path:
        source = Path(path)
        if source.suffix == "":
            source = source.with_suffix(suffix)
        elif source.suffix.lower() != suffix:
            raise SerializationError(f"path must use the {suffix} suffix")
        return source

    @staticmethod
    def _atomic_write(destination: Path, writer: Callable[[IO[bytes]], None]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise SerializationError(f"provenance.{field} must be a non-empty string")
    return value


def _raise_nonfinite_json(value: str) -> None:
    raise ValidationError(f"nonfinite JSON number {value!r} is not supported")


__all__ = ["ResultSerializer"]
