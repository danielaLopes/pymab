"""Strict validation helpers used at PyMAB's public boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real
from typing import cast

import numpy as np

from pymab.errors import ValidationError
from pymab.types import BoolArray, FloatArray, IntArray


def finite_float(value: object, *, name: str) -> float:
    """Return a finite real scalar without accepting booleans or strings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    """Return a built-in integer greater than or equal to ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValidationError(f"{name} must be {qualifier}")
    return result


def integer_array(
    value: object,
    *,
    name: str,
    ndim: int,
    allow_empty: bool = False,
    minimum: int | None = None,
    maximum_exclusive: int | None = None,
    readonly: bool = False,
) -> IntArray:
    """Validate integer-labelled data before converting it to ``int64``.

    Booleans, strings, and floats are rejected even when NumPy could coerce them
    losslessly. This keeps persisted and in-memory contracts identical.
    """

    objects = np.asarray(value, dtype=object)
    if objects.ndim != ndim or (not allow_empty and objects.size == 0):
        empty = "possibly empty" if allow_empty else "non-empty"
        raise ValidationError(f"{name} must be a {empty} {ndim}D array")
    for item in cast(Iterable[object], objects.flat):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValidationError(f"{name} must contain only integers")
    try:
        result = np.asarray(objects, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} contains an integer outside int64 range"
        ) from exc
    if minimum is not None and np.any(result < minimum):
        raise ValidationError(f"{name} contains a value below {minimum}")
    if maximum_exclusive is not None and np.any(result >= maximum_exclusive):
        raise ValidationError(
            f"{name} contains a value outside [0, {maximum_exclusive})"
        )
    if readonly:
        result = result.copy()
        result.flags.writeable = False
    return result


def float_array(
    value: object,
    *,
    name: str,
    ndim: int,
    allow_empty: bool = False,
    readonly: bool = False,
) -> FloatArray:
    """Validate numeric finite data before converting it to ``float64``."""

    objects = np.asarray(value, dtype=object)
    if objects.ndim != ndim or (not allow_empty and objects.size == 0):
        empty = "possibly empty" if allow_empty else "non-empty"
        raise ValidationError(f"{name} must be a {empty} {ndim}D array")
    for item in cast(Iterable[object], objects.flat):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValidationError(f"{name} must contain only real numbers")
    result = np.asarray(objects, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")
    if readonly:
        result = result.copy()
        result.flags.writeable = False
    return result


def boolean_array(
    value: object,
    *,
    name: str,
    ndim: int,
    allow_empty: bool = False,
    readonly: bool = False,
) -> BoolArray:
    """Validate boolean data without truthiness coercion."""

    objects = np.asarray(value, dtype=object)
    if objects.ndim != ndim or (not allow_empty and objects.size == 0):
        empty = "possibly empty" if allow_empty else "non-empty"
        raise ValidationError(f"{name} must be a {empty} {ndim}D array")
    if any(not isinstance(item, (bool, np.bool_)) for item in objects.flat):
        raise ValidationError(f"{name} must contain only booleans")
    result = np.asarray(objects, dtype=np.bool_)
    if readonly:
        result = result.copy()
        result.flags.writeable = False
    return result


def probability_vector(value: object, *, n_arms: int, name: str) -> FloatArray:
    """Return a validated probability vector with one value per arm."""

    result = float_array(value, name=name, ndim=1)
    if result.shape != (n_arms,):
        raise ValidationError(f"{name} must contain one probability per arm")
    if np.any(result < 0):
        raise ValidationError(f"{name} must be non-negative")
    if not np.isclose(np.sum(result), 1.0, rtol=1e-12, atol=1e-12):
        raise ValidationError(f"{name} must sum to one")
    return result


__all__ = [
    "boolean_array",
    "finite_float",
    "float_array",
    "integer_array",
    "positive_integer",
    "probability_vector",
]
