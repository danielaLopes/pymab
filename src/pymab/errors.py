"""PyMAB's public exception hierarchy."""

from __future__ import annotations


class PyMABError(Exception):
    """Base class for errors raised by PyMAB."""


class ValidationError(PyMABError, ValueError):
    """Raised when external data or configuration violates a contract."""


class CompatibilityError(PyMABError, TypeError):
    """Raised when two valid components cannot be used together."""


class OverlapError(ValidationError):
    """Raised when logged data cannot identify an off-policy estimate."""


class SerializationError(PyMABError):
    """Raised when a persisted result cannot be encoded or decoded safely."""


__all__ = [
    "CompatibilityError",
    "OverlapError",
    "PyMABError",
    "SerializationError",
    "ValidationError",
]
