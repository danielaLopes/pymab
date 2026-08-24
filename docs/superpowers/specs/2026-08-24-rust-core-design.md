# PyMAB Rust Core Design

**Date:** 2026-08-24

**Target release:** Unscheduled; version selected by the normal release process

**Status:** Revised; pending user review

## Summary

The next Rust-backed PyMAB release will move every built-in bandit policy and the
complete simulation hot loop into a published, pure Rust crate. A private PyO3
crate will expose that core to the existing Python package. The Python API will
remain the primary interface for analysis, plotting, persistence, and custom
extensions, while Rust users will be able to use the policy and simulation engine
directly.

The implementation must satisfy four outcomes:

1. Every concrete policy exported by `pymab.policies` has a Rust implementation.
2. Cross-backend tests demonstrate semantic and numerical parity with the Python
   reference implementation.
3. Reproducible benchmarks demonstrate lower execution time and lower incremental
   memory use for representative classic and contextual workloads.
4. One release process publishes matching versioned artifacts to crates.io and
   PyPI.

This design does not assign a release number. The version will be chosen when the
release is prepared, based on the versions actually published at that time and
the compatibility impact of the completed implementation. The Rust crate and
Python distribution will use the same selected version.

## Scope

### Rust policy implementations

The Rust core will implement all 27 concrete policies currently exported by
`pymab.policies`:

- `RandomPolicy`
- `GreedyPolicy`
- `EpsilonGreedyPolicy`
- `DecayingEpsilonGreedyPolicy`
- `SoftmaxPolicy`
- `UCBPolicy`
- `KLUCBPolicy`
- `MOSSPolicy`
- `GradientBanditPolicy`
- `BernoulliThompsonSamplingPolicy`
- `GaussianThompsonSamplingPolicy`
- `BernoulliBayesianUCBPolicy`
- `GaussianBayesianUCBPolicy`
- `EXP3Policy`
- `SuccessiveEliminationPolicy`
- `MedianEliminationPolicy`
- `SlidingWindowUCBPolicy`
- `DiscountedUCBPolicy`
- `SlidingWindowBernoulliThompsonSamplingPolicy`
- `DiscountedBernoulliThompsonSamplingPolicy`
- `ChangePointUCBPolicy`
- `CUSUMUCBPolicy`
- `PageHinkleyUCBPolicy`
- `LinearEpsilonGreedyPolicy`
- `LinUCBPolicy`
- `LinearThompsonSamplingPolicy`
- `LogisticContextualBanditPolicy`

The Rust core will also contain the policy traits, state types, validation rules,
environment dynamics, reward distributions, RNG stream derivation, experiment
runner, and result buffers needed to execute these policies without entering
Python during a simulation.

### Python responsibilities

The following remain in Python in the initial Rust-backed release:

- Statistical summaries, bootstrap analysis, and policy comparisons.
- Offline estimators and adaptive replay.
- Persistence schemas and provenance presentation.
- Plotting and pandas integrations.
- User-defined Python policy and environment extension points.

These components are outside the simulation hot loop and already use vectorized
NumPy operations or Python callback protocols. They may be ported later only with
separate evidence that a port improves user-visible performance.

## Repository and package layout

The existing repository becomes a Cargo workspace and a mixed Python/Rust
project:

```text
pymab/
├── Cargo.toml
├── crates/
│   ├── pymab-core/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   ├── tests/
│   │   └── benches/
│   └── pymab-python/
│       ├── Cargo.toml
│       └── src/
├── src/pymab/
├── tests/
├── benchmarks/
└── pyproject.toml
```

`crates/pymab-core` is published to crates.io with package and library name
`pymab`. It has no dependency on Python or PyO3. `crates/pymab-python` builds the
private extension module `pymab._pymab`, depends on `pymab-core` by path and
matching version, and has `publish = false`.

The workspace specifies a minimum supported Rust version of 1.83. The root
workspace version is the only release version. Both Rust packages inherit it,
and `pyproject.toml` declares its Python version as dynamic so Maturin derives it
from Cargo metadata.

## Rust core architecture

### Core traits

The non-contextual policy contract consists of configuration validation, action
selection, observation update, recommendation, reset, state inspection, and a
memory-footprint estimate. The contextual policy contract adds a context matrix
to selection, update, and recommendation. Concrete policies are represented by
typed structs for native Rust use.

The experiment runner uses internal policy enums to dispatch built-in policies.
This avoids one virtual call per decision while the public Rust API continues to
offer traits for downstream generic code. Contextual and non-contextual enums are
separate so invalid combinations are unrepresentable inside the runner.

Rust policy state uses `u64` for step and pull counts and `f64` for reward,
estimate, posterior, and matrix values. Public methods reject non-finite inputs.
Binary policies accept only exact `0.0` and `1.0`. Index errors and shape errors
return typed errors rather than panicking.

### Linear algebra and distributions

Contextual policies use a Rust linear-algebra dependency with owned, contiguous
matrices. Factorizations are reused where the algorithm allows it; explicit
matrix inversion is avoided. Cholesky or solve operations return typed numerical
errors when inputs cannot be factorized.

Random sampling uses `rand` and `rand_distr`. Bayesian UCB uses a maintained
statistical distribution implementation for Beta and Normal quantiles. Dependency
versions will be pinned in `Cargo.lock`, while the published library manifest uses
compatible SemVer requirements.

### Randomness and reproducibility

Rust uses a documented, seedable generator with stable stream derivation. The
same logical streams as the existing Python backend remain isolated:

- Environment dynamics.
- Context generation.
- Common potential rewards.
- Per-policy action selection.
- Per-policy independent rewards.

Stream seeds are derived from the master seed, replicate index, stream role, and
stable policy ID. Adding or reordering policies therefore does not alter another
policy's Rust stream.

Rust and NumPy are not required to produce bit-for-bit identical random samples.
Each backend is independently reproducible for a fixed version, configuration,
and seed. Provenance records the selected backend and RNG algorithm so stored
results are interpretable.

### Environments and experiment runner

The Rust runner owns the full nested replicate, step, and policy loop. It advances
the environment, generates contexts and potential rewards, selects actions,
updates policies, computes recommendations, and writes directly into contiguous
result buffers.

Result dimensions and public orientation remain identical to the existing Python
API. The binding transfers completed buffers into NumPy arrays at the experiment
boundary; it does not construct Python objects per step. The GIL is released for
the entire native run.

The first implementation is single-threaded to preserve a simple deterministic
contract and establish a trustworthy baseline. The internal replicate boundary
will allow later opt-in parallelism without changing public result ordering.

## Python integration and compatibility

### Policy wrappers

Existing public Python class names, constructor parameter names, properties,
method signatures, and `repr` behavior remain stable where practical. Built-in
policy instances own native Rust state through PyO3. Python-visible array state
is returned as a copy or a read-only view with documented ownership; callers
cannot create unsynchronized Rust and Python state by mutating an exposed buffer.

The abstract `Policy` and `ContextualPolicy` contracts remain available for
custom Python implementations. Built-in policies carry an internal native policy
descriptor that the Rust experiment runner can consume without serializing state
through dictionaries.

### Backend selection

`ExperimentConfig` adds a backend value with three modes:

- `"auto"` is the default. It selects Rust when the environment, reward model,
  dynamics, context provider, and every policy are native-compatible. Otherwise,
  it uses the Python reference runner.
- `"rust"` requires native-compatible components and raises a compatibility error
  that identifies every incompatible component.
- `"python"` always uses the reference runner.

The reference runner remains private but supported for custom Python components,
parity tests, migration diagnosis, and users whose target platform has no native
wheel and who build from source.

### Errors

The Rust crate defines non-panicking configuration, validation, numerical, and
compatibility error variants. PyO3 maps these to the existing PyMAB exception
hierarchy. Error messages retain the policy ID, replicate, step, offending field,
and expected range or shape where those details are currently available.

A panic at the FFI boundary is treated as an internal error and must not unwind
through Python. Tests exercise malformed configurations, invalid actions and
rewards, non-finite inputs, contextual shape errors, and numerical failures.

## Parity and correctness

### Parity definition

Parity means:

- Exact agreement for integer counters, selected deterministic actions, reset
  state, eliminated-arm masks, phase changes, and validation categories.
- Floating-point agreement within a documented absolute and relative tolerance
  for estimates, scores, probabilities, posterior parameters, change-detection
  statistics, and contextual matrices.
- Agreement on deterministic recommendations after a common observation trace.
- Agreement on the implied distribution for stochastic policies rather than
  equality between NumPy and Rust random samples.
- Independent seeded reproducibility for both backends.

The initial default tolerance is `rtol=1e-12, atol=1e-12`. Individual operations
may use a wider documented tolerance only when different, stable linear-algebra
implementations justify it.

### Shared fixtures

Each concrete policy has a data-driven parity fixture containing:

- Valid constructor configurations and boundary values.
- Invalid constructor configurations and expected error categories.
- A fixed action, reward, and optional context trace.
- Expected state after each update.
- Scores, probability vectors, posterior moments, or confidence indices.
- Final recommendation.
- Reset and clone expectations.

Fixtures are consumed by Python reference tests, Python native-wrapper tests, and
Rust integration tests. A coverage test compares the exported Python policy list,
the Rust built-in policy registry, and the fixture registry, failing if any set
differs.

Stochastic selection tests assert valid support, deterministic replay for a fixed
backend seed, and agreement with exposed probabilities or posterior parameters.
Large distributional checks are confined to dedicated tests with fixed sample
sizes and statistically conservative bounds so ordinary unit tests do not become
flaky.

### Additional test layers

- Rust unit tests cover formulae, boundary cases, and typed errors.
- `proptest` covers state invariants, finite outputs, probability normalization,
  reset behavior, and valid action ranges.
- Hypothesis continues to cover the Python public contract.
- The existing Python suite runs against `auto` and targeted cases run against
  both explicit backends.
- End-to-end tests cover result shapes, reward coupling, stream isolation,
  policy-order independence, context digests, provenance, and persistence
  round-trips.
- Rust documentation examples compile and Python README examples execute from a
  built wheel.

## Performance and memory evidence

### Benchmark suites

Rust Criterion benchmarks measure per-policy selection, update, and combined
decision throughput. They also measure complete classic and contextual experiment
runs without Python.

A separate-process cross-backend harness runs identical Python-facing workloads
with `backend="python"` and `backend="rust"`. Each case includes warm-up runs and
multiple measured repetitions. It records:

- Median elapsed time.
- Decisions per second.
- Process baseline RSS.
- Peak RSS and incremental peak RSS above the process baseline.
- Policy state bytes after an identical update trace.
- Python, NumPy, PyMAB, Rust, target, operating system, and CPU metadata.

The harness emits machine-readable JSON. A deterministic report generator turns
the JSON into a checked documentation table without hand-edited performance
claims.

### Representative workloads

The committed suite contains at least:

1. A stationary Gaussian classic experiment with several action-value policies.
2. A Bernoulli experiment with Thompson, Bayesian UCB, KL-UCB, and EXP3.
3. A non-stationary experiment with sliding-window, discounted, and
   change-detection policies.
4. A contextual experiment containing all four contextual policies.

Cases use enough replicates and horizon steps for execution time to dominate
process startup and binding overhead. Output sizes are reported so memory results
cannot hide the cost of returned arrays.

### Acceptance criteria

On the documented reference runner:

- Every benchmarked Rust workload must be faster than its Python counterpart.
- The aggregate classic and contextual suites must each be at least 2 times
  faster by median elapsed time.
- Every Rust built-in policy state must use fewer measured bytes than its Python
  reference state after the same trace.
- Aggregate classic and contextual workloads must use less incremental peak RSS
  than the Python backend.

Pull-request CI compiles benchmarks and performs short functional smoke runs.
Scheduled and manually dispatched CI run the full comparison on a fixed runner
class, enforce conservative relative thresholds, and upload raw JSON and the
rendered report. Relative same-run comparisons are authoritative; historic
absolute timings are informational because shared runner hardware can change.

## CI

The existing CI remains responsible for Python formatting, linting, security,
tests, documentation, examples, and package inspection. It gains the following
gates:

- `cargo fmt --all --check`.
- Clippy for the workspace with warnings denied.
- Rust unit, integration, property, and documentation tests.
- A Rust 1.83 MSRV build and test job.
- Rust dependency auditing.
- Cross-backend parity tests on Python 3.11, 3.12, 3.13, and 3.14.
- `cargo package` and `cargo publish --dry-run` for the public crate.
- Native wheel builds followed by isolated-environment import and execution
  smoke tests.
- Benchmark compilation and smoke execution.

All third-party actions are pinned to immutable commit SHAs. Cargo uses
`--locked` in CI and release builds. Build artifacts never inherit checkout
credentials.

## Packaging and release

### Versioning

The workspace package version is authoritative. The public core crate and private
binding crate inherit it. Maturin derives the Python distribution version from
Cargo. Release Please updates the workspace version, lock file, changelog, and
release manifest in one release pull request. A CI script asserts agreement among
Cargo metadata, built wheel metadata, Python `pymab.__version__`, documentation,
the git tag, and the `.crate` package.

### Python artifacts

Maturin builds:

- One source distribution containing the Python sources and both Rust crates.
- CPython 3.11 through 3.14 wheels for Linux x86-64 and aarch64.
- CPython 3.11 through 3.14 wheels for macOS x86-64 and arm64.
- CPython 3.11 through 3.14 wheels for Windows x86-64.

Linux wheels use an explicit manylinux policy and are audited for PyPI
compatibility. Each wheel is installed into an isolated environment on a matching
platform and must import PyMAB, report the expected version, identify the Rust
backend, and run a native experiment.

### Registry publication

A GitHub release triggers artifact construction and verification. No registry
publish job starts until the Rust package, source distribution, every wheel, and
their smoke tests have succeeded.

The public `pymab` crate is published first, followed by the Python distribution.
The release workflow checks whether the exact version already exists in each
registry before publishing, making a retry safe after a partial release. It never
overwrites an artifact because both registries treat a released version as
immutable.

PyPI uses GitHub OIDC trusted publishing. The first crates.io release uses a
scoped repository secret to establish crate ownership. After the repository is
registered as a crates.io trusted publisher, subsequent releases exchange a
GitHub OIDC token for a short-lived crates.io credential. The release environment
may require manual approval in GitHub.

Verified artifacts and the benchmark report are attached to the GitHub release.

## Migration and documentation

The Rust migration guide explains:

- Why seeded trajectories may differ from the existing Python backend.
- How to select `auto`, `rust`, or `python` explicitly.
- Which custom components force a Python fallback.
- How native wheels affect supported platforms and source builds.
- How to use the new Rust crate directly.
- The parity definition and how performance claims were measured.

Rust API documentation includes a minimal policy loop, a complete experiment,
custom policy trait usage, error handling, and reproducibility guarantees. Python
documentation keeps existing examples and adds backend selection and diagnostics.

## Implementation sequencing

Implementation will proceed in dependency order:

1. Create the Cargo workspace, core error/RNG/state contracts, PyO3 skeleton, and
   mixed-package development workflow.
2. Port action-value and stationary non-contextual policies with shared parity
   fixtures.
3. Port Bayesian, adversarial, pure-exploration, non-stationary, and
   change-detection policies.
4. Port contextual policies and linear algebra.
5. Port environments, reward sampling, and the complete experiment runner.
6. Replace built-in Python policy internals with native wrappers and implement
   backend selection and fallback.
7. Complete parity, property, integration, performance, and memory evidence.
8. Update documentation, CI, packaging, Release Please, and dual-registry release
   automation.
9. Run the full completion audit against every policy, artifact, platform, and
   acceptance criterion before declaring the migration complete.

Each stage must leave both the Rust workspace and supported Python test subset in
a working state. The Python reference backend is removed only by a future design;
it is deliberately retained in the initial Rust-backed release.
