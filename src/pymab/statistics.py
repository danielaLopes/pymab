"""Typed configuration and results for statistical uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

import numpy as np

from pymab.provenance import JSONValue
from pymab.types import FloatArray
from pymab.validation import finite_float, float_array, positive_integer


class ConfidenceMethod(StrEnum):
    """Confidence-interval construction methods supported by PyMAB."""

    PERCENTILE_BOOTSTRAP = "percentile_bootstrap"


class ResamplingUnit(StrEnum):
    """Independent unit resampled by an uncertainty calculation."""

    EVENT = "event"
    CLUSTER = "cluster"
    REPLICATE = "replicate"


@dataclass(frozen=True)
class BootstrapConfig:
    """Validated controls for deterministic, memory-bounded bootstrapping."""

    confidence_level: float = 0.95
    n_resamples: int = 10_000
    seed: int = 0
    max_chunk_elements: int = 1_000_000

    def __post_init__(self) -> None:
        confidence = finite_float(self.confidence_level, name="confidence_level")
        if not 0 < confidence < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "n_resamples",
            positive_integer(self.n_resamples, name="n_resamples"),
        )
        object.__setattr__(self, "seed", int(self.seed))
        budget = positive_integer(self.max_chunk_elements, name="max_chunk_elements")
        if budget < self.n_resamples:
            raise ValueError("max_chunk_elements must be at least n_resamples")
        object.__setattr__(self, "max_chunk_elements", budget)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible configuration record."""

        return {
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "max_chunk_elements": self.max_chunk_elements,
        }


@dataclass(frozen=True)
class IntervalEstimate:
    """A scalar estimate with optional bootstrap uncertainty."""

    estimate: float
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float
    confidence_method: ConfidenceMethod
    n_observations: int
    resampling_unit: ResamplingUnit

    def __post_init__(self) -> None:
        estimate = finite_float(self.estimate, name="estimate")
        confidence = finite_float(self.confidence_level, name="confidence_level")
        if not 0 < confidence < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        standard_error = self.standard_error
        if standard_error is not None:
            standard_error = finite_float(standard_error, name="standard_error")
            if standard_error < 0:
                raise ValueError("standard_error must be non-negative")
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("ci_lower and ci_upper must both be present or absent")
        lower = self.ci_lower
        upper = self.ci_upper
        if lower is not None and upper is not None:
            lower = finite_float(lower, name="ci_lower")
            upper = finite_float(upper, name="ci_upper")
            if lower > upper:
                raise ValueError("ci_lower must not exceed ci_upper")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "ci_lower", lower)
        object.__setattr__(self, "ci_upper", upper)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self, "confidence_method", ConfidenceMethod(self.confidence_method)
        )
        object.__setattr__(
            self,
            "n_observations",
            positive_integer(self.n_observations, name="n_observations"),
        )
        object.__setattr__(
            self, "resampling_unit", ResamplingUnit(self.resampling_unit)
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible estimate record."""

        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "confidence_method": self.confidence_method.value,
            "n_observations": self.n_observations,
            "resampling_unit": self.resampling_unit.value,
        }


def standard_error(values: object) -> float | None:
    """Return the standard error of a finite one-dimensional sample."""

    data = float_array(values, name="values", ndim=1)
    if data.size < 2:
        return None
    return float(np.std(data, ddof=1) / np.sqrt(data.size))


def bootstrap_mean_interval(
    values: object,
    *,
    config: BootstrapConfig | None = None,
) -> IntervalEstimate:
    """Return a deterministic percentile-bootstrap estimate of a sample mean."""

    from pymab._resampling import summarize_observations

    settings = BootstrapConfig() if config is None else config
    if not isinstance(settings, BootstrapConfig):
        raise TypeError("config must be a BootstrapConfig")
    data: FloatArray = float_array(values, name="values", ndim=1)
    return summarize_observations(
        data,
        config=settings,
        resampling_unit=ResamplingUnit.REPLICATE,
    )


__all__ = [
    "BootstrapConfig",
    "ConfidenceMethod",
    "IntervalEstimate",
    "ResamplingUnit",
    "bootstrap_mean_interval",
    "standard_error",
]
