from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.memory import deep_size
from benchmarks.report import evaluate_thresholds, render_rst
from benchmarks.run_backends import measure_case
from pymab import _native


def _payload() -> dict[str, object]:
    environment = {"python": "3.12", "rust_core": "2.0.0"}
    return {
        "schema_version": 1,
        "cases": [
            {
                "case": "stationary",
                "suite": "classic",
                "decisions": 100,
                "backends": {
                    "python": {
                        "median_elapsed_seconds": 4.0,
                        "median_incremental_peak_rss_bytes": 2_000,
                        "policy_state_bytes": {"greedy": 1_000},
                        "environment": environment,
                    },
                    "rust": {
                        "median_elapsed_seconds": 1.0,
                        "median_incremental_peak_rss_bytes": 500,
                        "policy_state_bytes": {"greedy": 100},
                        "environment": environment,
                    },
                },
            },
            {
                "case": "contextual",
                "suite": "contextual",
                "decisions": 50,
                "backends": {
                    "python": {
                        "median_elapsed_seconds": 6.0,
                        "median_incremental_peak_rss_bytes": 3_000,
                        "policy_state_bytes": {"greedy": 1_000},
                        "environment": environment,
                    },
                    "rust": {
                        "median_elapsed_seconds": 2.0,
                        "median_incremental_peak_rss_bytes": 1_000,
                        "policy_state_bytes": {"greedy": 100},
                        "environment": environment,
                    },
                },
            },
        ],
    }


def test_deep_size_counts_numpy_storage_and_cycles() -> None:
    array = np.zeros(20, dtype=np.float64)
    cyclic: list[object] = [array]
    cyclic.append(cyclic)
    assert deep_size(cyclic) >= array.nbytes


def test_threshold_evaluation_reports_regressions() -> None:
    thresholds = {
        "time": {
            "minimum_case_speedup": 1.0,
            "minimum_classic_speedup": 2.0,
            "minimum_contextual_speedup": 2.0,
        },
        "memory": {
            "maximum_policy_state_ratio": 1.0,
            "maximum_aggregate_incremental_rss_ratio": 1.0,
        },
    }
    assert evaluate_thresholds(_payload(), thresholds) == []
    payload = _payload()
    payload["cases"][0]["backends"]["rust"]["median_elapsed_seconds"] = 5.0  # type: ignore[index]
    failures = evaluate_thresholds(payload, thresholds)
    assert any("stationary" in failure for failure in failures)


def test_report_is_generated_only_from_json_values(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    rendered = render_rst(_payload(), source)
    assert "4.00x" in rendered
    assert str(source) in rendered
    assert "Measurement environment" in rendered


def test_isolated_backend_workers_share_output_contract() -> None:
    if not _native.native_available():
        pytest.skip("native extension has not been built")
    measurement = measure_case(
        "stationary",
        horizon=2,
        n_replicates=1,
        repetitions=1,
        sample_interval=0.001,
    )
    python = measurement["backends"]["python"]
    rust = measurement["backends"]["rust"]
    assert python["output_bytes"] == rust["output_bytes"]
    assert set(python["policy_state_bytes"]) == set(rust["policy_state_bytes"])
    assert len(rust["policy_state_bytes"]) == 27
