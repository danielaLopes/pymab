"""Child-process worker for one backend and one canonical workload."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

import pymab
from benchmarks.cases import CASE_NAMES, build_experiment
from benchmarks.memory import Backend, measure_policy_states
from pymab.results import SimulationResult


class WorkerResult(TypedDict):
    """JSON contract written by one isolated worker."""

    backend: Backend
    case: str
    decisions: int
    elapsed_seconds: float
    environment: dict[str, str]
    n_policies: int
    output_bytes: int
    policy_state_bytes: dict[str, int]


def result_bytes(result: SimulationResult) -> int:
    """Return bytes owned by result arrays."""

    arrays = (
        result.rewards,
        result.actions,
        result.expected_rewards,
        result.arm_means,
        result.optimal_mask,
        result.recommendations,
    )
    total = sum(int(array.nbytes) for array in arrays)
    if result.contexts is not None:
        total += int(result.contexts.nbytes)
    return total


def environment_metadata() -> dict[str, str]:
    """Capture runtime identity needed to interpret a measurement."""

    return {
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "pymab": pymab.__version__,
        "python": platform.python_version(),
        "rust_core": pymab._native.core_version() or "unavailable",
    }


def run_worker(
    *,
    case: str,
    backend: Backend,
    horizon: int,
    n_replicates: int,
) -> WorkerResult:
    """Build, synchronize with the parent, and measure one complete run."""

    experiment = build_experiment(
        case,
        horizon=horizon,
        n_replicates=n_replicates,
        backend=backend,
    )
    state_bytes = measure_policy_states(backend)
    print("READY", flush=True)
    if not sys.stdin.readline():
        raise RuntimeError("benchmark parent disconnected before measurement")
    started = time.perf_counter()
    result = experiment.run()
    elapsed = time.perf_counter() - started
    n_policies = len(experiment.policies)
    return {
        "backend": backend,
        "case": case,
        "decisions": horizon * n_replicates * n_policies,
        "elapsed_seconds": elapsed,
        "environment": environment_metadata(),
        "n_policies": n_policies,
        "output_bytes": result_bytes(result),
        "policy_state_bytes": state_bytes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--backend", choices=("python", "rust"), required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--n-replicates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synchronized worker and atomically replace its result file."""

    arguments = _parser().parse_args(argv)
    output = cast(Path, arguments.output)
    measurement = run_worker(
        case=cast(str, arguments.case),
        backend=cast(Backend, arguments.backend),
        horizon=arguments.horizon,
        n_replicates=arguments.n_replicates,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(measurement, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = ["WorkerResult", "environment_metadata", "main", "result_bytes", "run_worker"]
