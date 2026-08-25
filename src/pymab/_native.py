"""Optional access to the compiled PyMAB extension."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def _load_extension() -> ModuleType | None:
    try:
        return import_module("pymab._pymab")
    except ModuleNotFoundError as exc:
        if exc.name != "pymab._pymab":
            raise
        return None


_EXTENSION = _load_extension()


def native_available() -> bool:
    """Return whether the optional native extension is importable."""

    return _EXTENSION is not None and bool(_EXTENSION.native_available())


def core_version() -> str | None:
    """Return the linked Rust core version, or ``None`` when unavailable."""

    if _EXTENSION is None:
        return None
    return str(_EXTENSION.core_version())


def rng_scheme_version() -> str | None:
    """Return the native RNG scheme identifier, or ``None`` when unavailable."""

    if _EXTENSION is None:
        return None
    return str(_EXTENSION.rng_scheme_version())


def create_policy(kind: str, configuration_json: str) -> Any:
    """Construct a private native policy handle.

    This internal function is intentionally absent from the public package API;
    public policy wrappers provide the stable typed constructors.
    """

    if _EXTENSION is None:
        raise RuntimeError("the optional PyMAB native extension is unavailable")
    return _EXTENSION._NativePolicy.create(kind, configuration_json)


def create_environment(configuration_json: str) -> Any:
    """Construct a private native environment handle."""

    if _EXTENSION is None:
        raise RuntimeError("the optional PyMAB native extension is unavailable")
    return _EXTENSION._NativeEnvironment.create(configuration_json)


def run_experiment(
    environment: Any,
    policies: list[tuple[str, Any]],
    horizon: int,
    n_replicates: int,
    seed: int,
    reward_coupling: str,
    record_contexts: bool,
) -> Any:
    """Run a private native experiment and return NumPy-owned buffers."""

    if _EXTENSION is None:
        raise RuntimeError("the optional PyMAB native extension is unavailable")
    return _EXTENSION._NativeExperiment.run(
        environment,
        policies,
        horizon,
        n_replicates,
        seed,
        reward_coupling,
        record_contexts,
    )


__all__ = ["core_version", "native_available", "rng_scheme_version"]
