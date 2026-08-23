# PyMAB Statistical Core Design

Date: 2026-08-23
Status: Approved for implementation
Target branch: `improve-statistical-analysis`

## Purpose

Consolidate PyMAB's three bootstrap implementations into one deterministic,
memory-bounded statistical core. Replace unstructured benchmarking dictionaries
with typed immutable results, simplify configuration, and strengthen reference
tests for uncertainty estimates used by data scientists.

This branch changes the unreleased v2 API where doing so produces a clearer and
more maintainable contract. It does not add bandit algorithms or change the
simulation architecture.

## Public API

Add `pymab.statistics` with these public types:

- `BootstrapConfig`: immutable configuration containing `confidence_level`,
  `n_resamples`, `seed`, and `max_chunk_elements`.
- `IntervalEstimate`: immutable scalar estimate containing the point estimate,
  standard error, interval bounds, confidence level, confidence method,
  observation count, and resampling unit.
- `bootstrap_mean_interval`: deterministic percentile-bootstrap analysis for a
  one-dimensional sample.

Add typed benchmarking records:

- `BenchmarkConfig`: baseline and bootstrap configuration.
- `PolicySummary`: one policy's typed metric estimates.
- `PolicyComparison`: typed paired differences against a baseline.

`BenchmarkResult.summary()` returns `tuple[PolicySummary, ...]` and
`compare_to_baseline()` returns `tuple[PolicyComparison, ...]`. `to_dict()`
remains the JSON-compatible boundary, and `to_pandas()` is built from the typed
records.

`compare()` continues to accept the experiment `config` and adds an `analysis`
configuration instead of separate analysis keyword arguments.

`EstimatorConfig` contains a `BootstrapConfig`. `estimate_policy_value()`
accepts an estimator `config` plus an optional reward model instead of separate
method, clipping, confidence, resample, budget, and seed keywords.

Plotting accepts `BootstrapConfig` for confidence bands. The specialized
`BootstrapBandConfig` is removed. `bootstrap_mean_interval` moves from
`pymab.benchmarking` to `pymab.statistics` so it has one stable import path.

## Shared resampling engine

An internal `_resampling` module owns validation, seeded random sampling,
chunking, standard errors, and percentile quantiles. It supports:

- scalar means and paired differences;
- self-normalized ratios used by SNIPS;
- event bootstrap;
- whole-cluster bootstrap;
- replicate-level curve bands.

Chunking must not change results for a fixed seed. The engine interprets
`max_chunk_elements` as a hard upper bound on its largest resampling workspace.
If even one complete statistical unit cannot fit, it raises `ValidationError`
instead of silently exceeding the budget.

Cluster bootstrap aggregates contribution sums, weight sums, and row counts by
cluster before resampling. It samples fixed-size cluster-index chunks and never
constructs concatenated event-index arrays, including for highly unequal
cluster sizes.

Percentile bootstrap remains the sole interval method. Cases with fewer than
two independent units return a point estimate with `None` uncertainty fields.
Nonfinite results are excluded only when the statistic is mathematically
undefined for a resample; fewer than two finite resamples produces unavailable
uncertainty rather than a misleading interval.

## Statistical behavior

Offline estimation preserves current IPS, SNIPS, and doubly robust point
estimators, overlap failures, clipping diagnostics, and event-versus-cluster
semantics. The common engine changes only uncertainty calculation and memory
behavior.

Benchmarking continues to aggregate within each replicate before calculating
uncertainty. Baseline comparisons remain paired by replicate. Plot bands
resample replicate indices consistently across all steps and policies.

Every random analysis remains deterministic and isolated from experiment RNG
streams. General analysis defaults to 10,000 resamples; plotting uses an
explicit 2,000-resample default configuration to preserve interactive cost.

## Testing and completion criteria

Tests must cover:

1. Strict `BootstrapConfig` validation, including booleans, nonfinite values,
   invalid confidence levels, and insufficient budgets.
2. Event, paired, ratio, cluster, and curve results against small brute-force or
   fixed-seed reference calculations.
3. Bit-for-bit chunk-budget invariance for all resampling modes.
4. Unequal and repeated clusters, one independent unit, zero overlap, clipped
   weights, and resamples with zero SNIPS denominators.
5. Seeded Monte Carlo bias and interval-coverage checks for IPS, SNIPS, doubly
   robust estimation, and paired policy comparisons.
6. Typed summary/comparison records, JSON conversion, pandas conversion, and
   plotting integration.
7. Instrumented allocation requests proving the workspace budget is honored.
8. Migration examples for every changed public signature or import.

The branch must reach at least 95% branch coverage across the new statistics,
resampling, benchmarking, plotting, and offline-estimation modules and raise the
repository-wide threshold to 92%. Ruff, strict mypy, Bandit, dependency audit,
Python 3.11 through 3.14, minimum dependencies, strict Sphinx builds, examples,
source distributions, wheels, Twine validation, and installed-wheel smoke/type
checks must all pass.

NumPy remains the only required runtime dependency. Pandas and Plotly stay
lazy, optional integrations. Unrelated plan files and generated artifacts are
excluded from every commit.
