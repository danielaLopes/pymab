"""Measure one canonical workload using the Python reference backend."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from typing import TypedDict, cast

import numpy as np

import pymab
from benchmarks.cases import CASE_NAMES, build_experiment, case_defaults
from pymab.results import SimulationResult


class EnvironmentMetadata(TypedDict):
    """Runtime metadata attached to a benchmark measurement."""

    implementation: str
    machine: str
    numpy: str
    platform: str
    processor: str
    pymab: str
    python: str


class ReferenceMeasurement(TypedDict):
    """Machine-readable reference benchmark result."""

    schema_version: int
    backend: str
    case: str
    horizon: int
    n_replicates: int
    n_policies: int
    decisions: int
    result_bytes: int
    elapsed_seconds: list[float]
    median_elapsed_seconds: float
    decisions_per_second: float
    environment: EnvironmentMetadata


def measure_case(
    name: str,
    *,
    horizon: int,
    n_replicates: int,
    repetitions: int,
) -> ReferenceMeasurement:
    """Measure a canonical workload and return stable JSON-compatible data.

    Args:
        name: Registered benchmark case name.
        horizon: Number of decisions per replicate.
        n_replicates: Number of independent replicates.
        repetitions: Number of complete measured executions.

    Raises:
        TypeError: If ``repetitions`` is not an integer.
        ValueError: If ``repetitions`` is not positive or the case is unknown.
    """

    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    elapsed_seconds: list[float] = []
    result: SimulationResult | None = None
    n_policies = 0
    for _ in range(repetitions):
        experiment = build_experiment(
            name,
            horizon=horizon,
            n_replicates=n_replicates,
        )
        n_policies = len(experiment.policies)
        started = time.perf_counter()
        result = experiment.run()
        elapsed_seconds.append(time.perf_counter() - started)

    if result is None:  # pragma: no cover - guarded by repetitions validation
        raise RuntimeError("benchmark completed without a result")
    decisions = horizon * n_replicates * n_policies
    median_elapsed = float(statistics.median(elapsed_seconds))
    return {
        "schema_version": 1,
        "backend": "python-reference",
        "case": name,
        "horizon": horizon,
        "n_replicates": n_replicates,
        "n_policies": n_policies,
        "decisions": decisions,
        "result_bytes": _result_bytes(result),
        "elapsed_seconds": elapsed_seconds,
        "median_elapsed_seconds": median_elapsed,
        "decisions_per_second": decisions / median_elapsed,
        "environment": _environment_metadata(),
    }


def _result_bytes(result: SimulationResult) -> int:
    arrays = (
        result.rewards,
        result.actions,
        result.expected_rewards,
        result.arm_means,
        result.optimal_mask,
        result.recommendations,
    )
    total = sum(array.nbytes for array in arrays)
    if result.contexts is not None:
        total += result.contexts.nbytes
    return int(total)


def _environment_metadata() -> EnvironmentMetadata:
    return {
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pymab": pymab.__version__,
        "python": platform.python_version(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--n-replicates", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line worker and write one JSON object to stdout."""

    arguments = _parser().parse_args(argv)
    defaults = case_defaults(arguments.case)
    measurement = measure_case(
        cast(str, arguments.case),
        horizon=(defaults.horizon if arguments.horizon is None else arguments.horizon),
        n_replicates=(
            defaults.n_replicates
            if arguments.n_replicates is None
            else arguments.n_replicates
        ),
        repetitions=arguments.repetitions,
    )
    json.dump(measurement, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module execution
    raise SystemExit(main())


__all__ = ["ReferenceMeasurement", "main", "measure_case"]
