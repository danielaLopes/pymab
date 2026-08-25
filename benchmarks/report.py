"""Validate benchmark thresholds and render performance documentation from JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

DEFAULT_THRESHOLDS = Path(__file__).with_name("thresholds.toml")
DEFAULT_OUTPUT = Path(__file__).parents[1] / "docs" / "source" / "performance.rst"


def evaluate_thresholds(
    payload: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> list[str]:
    """Return human-readable threshold failures; an empty list means success."""

    failures: list[str] = []
    cases = cast(list[Mapping[str, Any]], payload["cases"])
    time_limits = cast(Mapping[str, float], thresholds["time"])
    memory_limits = cast(Mapping[str, float], thresholds["memory"])
    for case in cases:
        backends = cast(Mapping[str, Mapping[str, Any]], case["backends"])
        speedup = float(backends["python"]["median_elapsed_seconds"]) / float(
            backends["rust"]["median_elapsed_seconds"]
        )
        if speedup <= float(time_limits["minimum_case_speedup"]):
            failures.append(
                f"{case['case']}: Rust speedup {speedup:.2f}x is not above threshold"
            )
    for suite in ("classic", "contextual"):
        selected = [case for case in cases if case["suite"] == suite]
        if not selected:
            continue
        python_time = sum(
            float(case["backends"]["python"]["median_elapsed_seconds"])
            for case in selected
        )
        rust_time = sum(
            float(case["backends"]["rust"]["median_elapsed_seconds"])
            for case in selected
        )
        speedup = python_time / rust_time
        limit = float(time_limits[f"minimum_{suite}_speedup"])
        if speedup < limit:
            failures.append(
                f"{suite}: aggregate speedup {speedup:.2f}x is below {limit:.2f}x"
            )

    if cases:
        backends = cast(Mapping[str, Mapping[str, Any]], cases[0]["backends"])
        python_states = cast(
            Mapping[str, int], backends["python"]["policy_state_bytes"]
        )
        rust_states = cast(Mapping[str, int], backends["rust"]["policy_state_bytes"])
        maximum_ratio = float(memory_limits["maximum_policy_state_ratio"])
        for policy in sorted(python_states):
            ratio = rust_states[policy] / python_states[policy]
            if ratio >= maximum_ratio:
                failures.append(
                    f"{policy}: native state ratio {ratio:.3f} is not below {maximum_ratio:.3f}"
                )
        python_rss = sum(
            float(case["backends"]["python"]["median_incremental_peak_rss_bytes"])
            for case in cases
        )
        rust_rss = sum(
            float(case["backends"]["rust"]["median_incremental_peak_rss_bytes"])
            for case in cases
        )
        rss_ratio = rust_rss / python_rss if python_rss else float("inf")
        rss_limit = float(memory_limits["maximum_aggregate_incremental_rss_ratio"])
        if rss_ratio >= rss_limit:
            failures.append(
                f"aggregate incremental RSS ratio {rss_ratio:.3f} is not below {rss_limit:.3f}"
            )
    return failures


def render_rst(payload: Mapping[str, Any], source: Path) -> str:
    """Render an RST report containing only values derived from raw JSON."""

    lines = [
        "Native backend performance",
        "==========================",
        "",
        "These measurements compare isolated Python-reference and release-mode Rust",
        "workers on the same machine.",
        f"The report is generated from ``{source}``; timings",
        "are medians and memory is sampled child-process RSS after imports.",
        "",
        ".. list-table:: Runtime by canonical workload",
        "   :header-rows: 1",
        "",
        "   * - Case",
        "     - Decisions",
        "     - Python (s)",
        "     - Rust (s)",
        "     - Speedup",
    ]
    cases = cast(list[Mapping[str, Any]], payload["cases"])
    for case in cases:
        python = case["backends"]["python"]
        rust = case["backends"]["rust"]
        speedup = python["median_elapsed_seconds"] / rust["median_elapsed_seconds"]
        lines.extend(
            [
                f"   * - {case['case']}",
                f"     - {case['decisions']:,}",
                f"     - {python['median_elapsed_seconds']:.4f}",
                f"     - {rust['median_elapsed_seconds']:.4f}",
                f"     - {speedup:.2f}x",
            ]
        )
    classic = [case for case in cases if case["suite"] == "classic"]
    contextual = [case for case in cases if case["suite"] == "contextual"]

    def aggregate_speedup(selected: list[Mapping[str, Any]]) -> float:
        python = sum(
            case["backends"]["python"]["median_elapsed_seconds"] for case in selected
        )
        rust = sum(
            case["backends"]["rust"]["median_elapsed_seconds"] for case in selected
        )
        return float(python / rust)

    python_rss = sum(
        case["backends"]["python"]["median_incremental_peak_rss_bytes"]
        for case in cases
    )
    rust_rss = sum(
        case["backends"]["rust"]["median_incremental_peak_rss_bytes"] for case in cases
    )
    aggregate_lines = []
    if classic:
        aggregate_lines.append(
            f"* Classic runtime speedup: **{aggregate_speedup(classic):.2f}x**"
        )
    if contextual:
        aggregate_lines.append(
            f"* Contextual runtime speedup: **{aggregate_speedup(contextual):.2f}x**"
        )
    if python_rss:
        aggregate_lines.append(
            f"* Incremental peak RSS ratio (Rust/Python): **{rust_rss / python_rss:.3f}**"
        )
    lines.extend(
        [
            "",
            "Aggregate results",
            "-----------------",
            "",
            *aggregate_lines,
            "",
            "Memory evidence",
            "---------------",
            "",
            "State memory is capacity-aware on the Rust side and recursively measured",
            "for the private Python reference objects after identical shared-fixture traces.",
            "Incremental RSS excludes each worker's post-import baseline.",
            "",
            ".. list-table:: Incremental peak RSS",
            "   :header-rows: 1",
            "",
            "   * - Case",
            "     - Python (MiB)",
            "     - Rust (MiB)",
        ]
    )
    for case in cases:
        python = case["backends"]["python"]["median_incremental_peak_rss_bytes"] / 2**20
        rust = case["backends"]["rust"]["median_incremental_peak_rss_bytes"] / 2**20
        lines.extend(
            [f"   * - {case['case']}", f"     - {python:.2f}", f"     - {rust:.2f}"]
        )
    if cases:
        python_states = cases[0]["backends"]["python"]["policy_state_bytes"]
        rust_states = cases[0]["backends"]["rust"]["policy_state_bytes"]
        lines.extend(
            [
                "",
                ".. list-table:: Policy state after shared parity traces",
                "   :header-rows: 1",
                "",
                "   * - Policy",
                "     - Python (bytes)",
                "     - Rust (bytes)",
                "     - Rust/Python",
            ]
        )
        for policy in sorted(python_states):
            ratio = rust_states[policy] / python_states[policy]
            lines.extend(
                [
                    f"   * - {policy}",
                    f"     - {python_states[policy]:,}",
                    f"     - {rust_states[policy]:,}",
                    f"     - {ratio:.3f}",
                ]
            )
    environment = cases[0]["backends"]["rust"]["environment"] if cases else {}
    lines.extend(
        [
            "",
            "Measurement environment",
            "-----------------------",
            "",
            *(f"* **{key}:** {value}" for key, value in sorted(environment.items())),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--check-thresholds", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    thresholds = tomllib.loads(arguments.thresholds.read_text(encoding="utf-8"))
    failures = evaluate_thresholds(payload, thresholds)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_rst(payload, arguments.input), encoding="utf-8")
    if arguments.check_thresholds and failures:
        for failure in failures:
            print(f"threshold failure: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["evaluate_thresholds", "main", "render_rst"]
