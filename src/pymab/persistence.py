"""Atomic, schema-aware persistence for simulation results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, cast

import numpy as np

from pymab._result_schema import ARRAY_FIELDS, _metadata_payload, _result_from_payload
from pymab.errors import SerializationError, ValidationError
from pymab.results import SimulationResult


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
        metadata = _metadata_payload(result)

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
                for key in ARRAY_FIELDS:
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
        return _result_from_payload(payload)

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


def _raise_nonfinite_json(value: str) -> None:
    raise ValidationError(f"nonfinite JSON number {value!r} is not supported")


__all__ = ["ResultSerializer"]
