"""Run same-machine Python/Rust performance comparisons in isolated children."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psutil

from benchmarks.cases import CASE_NAMES, case_defaults
from benchmarks.memory import Backend


def _worker_once(
    *,
    case: str,
    backend: Backend,
    horizon: int,
    n_replicates: int,
    sample_interval: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pymab-benchmark-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.worker",
            "--case",
            case,
            "--backend",
            backend,
            "--horizon",
            str(horizon),
            "--n-replicates",
            str(n_replicates),
            "--output",
            str(output),
        ]
        process = subprocess.Popen(  # noqa: S603 - fixed local module invocation
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdout is None or process.stdin is None:
            process.kill()
            raise RuntimeError("benchmark worker pipes were not created")
        ready = process.stdout.readline().strip()
        if ready != "READY":
            _, errors = process.communicate(timeout=30)
            raise RuntimeError(
                f"benchmark worker did not become ready ({ready!r}): {errors}"
            )
        observed = psutil.Process(process.pid)
        baseline = observed.memory_info().rss
        peak = baseline
        process.stdin.write("run\n")
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        while process.poll() is None:
            try:
                peak = max(peak, observed.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(sample_interval)
        _, errors = process.communicate(timeout=30)
        if process.returncode != 0:
            raise RuntimeError(
                f"benchmark worker failed for {case}/{backend}: {errors}"
            )
        payload = cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))
        payload["baseline_rss_bytes"] = baseline
        payload["peak_rss_bytes"] = peak
        payload["incremental_peak_rss_bytes"] = max(0, peak - baseline)
        return payload


def measure_case(
    case: str,
    *,
    horizon: int,
    n_replicates: int,
    repetitions: int,
    sample_interval: float = 0.005,
) -> dict[str, Any]:
    """Measure both backends for one workload."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    backends: dict[str, dict[str, Any]] = {}
    for backend in cast(tuple[Backend, ...], ("python", "rust")):
        samples = [
            _worker_once(
                case=case,
                backend=backend,
                horizon=horizon,
                n_replicates=n_replicates,
                sample_interval=sample_interval,
            )
            for _ in range(repetitions)
        ]
        elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
        baseline = [int(sample["baseline_rss_bytes"]) for sample in samples]
        peak = [int(sample["peak_rss_bytes"]) for sample in samples]
        incremental = [int(sample["incremental_peak_rss_bytes"]) for sample in samples]
        median_elapsed = float(statistics.median(elapsed))
        representative = samples[0]
        backends[backend] = {
            "baseline_rss_bytes": baseline,
            "decisions_per_second": int(representative["decisions"]) / median_elapsed,
            "elapsed_seconds": elapsed,
            "environment": representative["environment"],
            "incremental_peak_rss_bytes": incremental,
            "median_baseline_rss_bytes": float(statistics.median(baseline)),
            "median_elapsed_seconds": median_elapsed,
            "median_incremental_peak_rss_bytes": float(statistics.median(incremental)),
            "median_peak_rss_bytes": float(statistics.median(peak)),
            "output_bytes": representative["output_bytes"],
            "peak_rss_bytes": peak,
            "policy_state_bytes": representative["policy_state_bytes"],
        }
    if backends["python"]["output_bytes"] != backends["rust"]["output_bytes"]:
        raise RuntimeError("backends produced differently sized result buffers")
    return {
        "backends": backends,
        "case": case,
        "decisions": samples[0]["decisions"],
        "horizon": horizon,
        "n_policies": samples[0]["n_policies"],
        "n_replicates": n_replicates,
        "suite": "contextual" if case == "contextual" else "classic",
    }


def run_suite(
    cases: Sequence[str],
    *,
    repetitions: int,
    horizon: int | None = None,
    n_replicates: int | None = None,
    sample_interval: float = 0.005,
) -> dict[str, Any]:
    """Measure selected cases and return the complete JSON document."""

    measurements = []
    for case in cases:
        defaults = case_defaults(case)
        measurements.append(
            measure_case(
                case,
                horizon=defaults.horizon if horizon is None else horizon,
                n_replicates=(
                    defaults.n_replicates if n_replicates is None else n_replicates
                ),
                repetitions=repetitions,
                sample_interval=sample_interval,
            )
        )
    return {
        "cases": measurements,
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_interval_seconds": sample_interval,
        "schema_version": 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--n-replicates", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    cases = CASE_NAMES if arguments.all else tuple(arguments.case)
    payload = run_suite(
        cases,
        repetitions=arguments.repetitions,
        horizon=arguments.horizon,
        n_replicates=arguments.n_replicates,
        sample_interval=arguments.sample_interval,
    )
    output = cast(Path, arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["main", "measure_case", "run_suite"]
