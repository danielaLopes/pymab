# PyMAB Rust Core Implementation Plan

**Date:** 2026-08-24

**Branch:** `rust-pymab`

**Design:**
[`docs/superpowers/specs/2026-08-24-rust-core-design.md`](../specs/2026-08-24-rust-core-design.md)

**Release version:** Not selected by this plan. Workspace metadata initially
mirrors the repository's current development version; the normal release process
chooses the eventual release version.

## Objective

Publish a pure Rust `pymab` crate, use it as the native engine for the Python
package through PyO3, port all 27 concrete exported policies, prove parity with
the retained Python reference backend, demonstrate execution-time and memory
improvements, and automate verified publication to crates.io and PyPI.

## Working rules

- Keep commits scoped to the task boundaries below.
- Start each behavior change with a failing Rust or Python test.
- Run focused tests during each task and the full gates at phase boundaries.
- Do not modify or commit `.superpowers/`, `BOT_DETECTION_PLAN.md`, or
  `TOR_PATH_SELECTOR_PLAN.md`; they are unrelated user files.
- Do not select or hard-code a future release number.
- Do not remove the Python reference backend in this migration.
- Do not claim a speed or memory improvement until the checked benchmark harness
  produces evidence on the same machine for both backends.

## Phase 1: Establish the mixed Rust/Python workspace

### Task 1: Capture the Python reference baseline

**Create**

- `benchmarks/__init__.py`
- `benchmarks/cases.py`
- `benchmarks/reference_worker.py`
- `tests/test_backend_contract.py`

**Modify**

- `.gitignore`

**Work**

1. Add a backend-contract test that records the current `ExperimentConfig`
   fields, result array shapes/dtypes, and current provenance keys.
2. Define four canonical workloads in `benchmarks/cases.py`: stationary classic,
   Bernoulli, non-stationary, and contextual.
3. Add a reference worker that can run one workload in a clean process and emit
   JSON containing elapsed time, decision count, output bytes, and environment
   metadata. Memory sampling is added later; this task establishes stable inputs.
4. Run the current Python suite and save command output in the implementation
   notes, not as hand-edited benchmark claims.

**Verify**

```bash
uv run pytest tests/test_backend_contract.py
uv run pytest --no-cov
uv run python -m benchmarks.reference_worker --case stationary --repetitions 1
```

**Commit**

```text
test: capture Python backend contract
```

### Task 2: Create the Cargo workspace and Maturin skeleton

**Create**

- `Cargo.toml`
- `Cargo.lock`
- `crates/pymab-core/Cargo.toml`
- `crates/pymab-core/src/lib.rs`
- `crates/pymab-python/Cargo.toml`
- `crates/pymab-python/src/lib.rs`
- `src/pymab/_native.py`
- `tests/test_native_import.py`

**Modify**

- `pyproject.toml`
- `uv.lock`
- `Makefile`
- `.gitignore`

**Work**

1. Create a resolver-v2 workspace with `pymab-core` and unpublished
   `pymab-python` members. The core package is published under the crate name
   `pymab`; its directory name remains `pymab-core`.
2. Put shared authors, license, repository, edition, MSRV 1.83, and the current
   development version in `[workspace.package]`.
3. Give the binding crate both `cdylib` and `rlib` outputs and expose a minimal
   `_pymab` module with `native_available()`, `core_version()`, and
   `rng_scheme_version()`.
4. Switch the Python build backend from setuptools to Maturin. Configure
   `python-source = "src"`, `module-name = "pymab._pymab"`, and a manifest path
   to the binding crate. Mark the Python project version dynamic so Cargo metadata
   is authoritative without selecting a new release version.
5. Add compatible PyO3, rust-numpy, Maturin, and Rust test dependencies. Lock
   exact resolved versions in `Cargo.lock` and `uv.lock`.
6. Add `make native`, `make rust-format`, `make rust-lint`, and `make rust-test`.
7. Test both editable native installation and isolated wheel installation.

**Verify**

```bash
cargo fmt --all --check
cargo test --workspace --locked
uv run maturin develop --manifest-path crates/pymab-python/Cargo.toml
uv run pytest tests/test_native_import.py
uv run maturin build --manifest-path crates/pymab-python/Cargo.toml --release
```

**Commit**

```text
build: establish mixed Rust Python workspace
```

### Task 3: Implement core errors, types, validation, and RNG

**Create**

- `crates/pymab-core/src/error.rs`
- `crates/pymab-core/src/types.rs`
- `crates/pymab-core/src/validation.rs`
- `crates/pymab-core/src/rng.rs`
- `crates/pymab-core/tests/rng_contract.rs`
- `crates/pymab-core/tests/validation_contract.rs`

**Modify**

- `crates/pymab-core/src/lib.rs`
- `crates/pymab-core/Cargo.toml`
- `src/pymab/provenance.py`
- `tests/test_simulation_reliability.py`

**Work**

1. Add typed configuration, validation, compatibility, numerical, and internal
   errors; public Rust APIs return `Result` and do not panic for user input.
2. Add `RewardDomain`, policy capability metadata, context shapes, action indices,
   and checked finite-number helpers.
3. Port stable labeled stream derivation using BLAKE2b and define a new explicit
   Rust RNG scheme identifier. Use a seedable ChaCha generator for native
   sampling.
4. Test stream stability, policy-ID isolation, policy-order independence, and
   distinct stream roles.
5. Extend provenance to include `backend` and `rng_scheme` while remaining able
   to read results created before those fields existed.

**Verify**

```bash
cargo test -p pymab --test rng_contract --locked
cargo test -p pymab --test validation_contract --locked
uv run pytest tests/test_simulation_reliability.py -k "provenance or seed or order"
```

**Commit**

```text
feat: add Rust core contracts and deterministic streams
```

## Phase 2: Build the policy core with shared parity fixtures

### Task 4: Create policy traits, state types, and fixture infrastructure

**Create**

- `crates/pymab-core/src/policy/mod.rs`
- `crates/pymab-core/src/policy/action_value.rs`
- `crates/pymab-core/src/policy/registry.rs`
- `crates/pymab-core/tests/policy_fixtures.rs`
- `tests/fixtures/policies/schema.json`
- `tests/fixtures/policies/registry.json`
- `tests/parity/__init__.py`
- `tests/parity/conftest.py`
- `tests/parity/test_fixture_coverage.py`
- `tests/parity/test_policy_state.py`

**Modify**

- `crates/pymab-core/Cargo.toml`
- `crates/pymab-core/src/lib.rs`
- `src/pymab/policies/policy.py`

**Work**

1. Add public `Policy` and `ContextualPolicy` Rust traits with select, update,
   recommend, reset, state inspection, and estimated state-byte methods.
2. Add internal classic/contextual built-in enums for monomorphic experiment
   dispatch.
3. Implement common action-value state, stable softmax, uniform tie-breaking,
   action validation, finite reward validation, clone/reset, and capacity-aware
   memory accounting.
4. Define JSON parity fixtures with policy kind, config, update trace, checkpoints,
   derived scores, recommendation, reset state, and expected error code.
5. Make both Rust and Python fixture loaders reject unknown or incomplete fixture
   fields.
6. Add a coverage test that compares all 27 names in the Python export registry,
   Rust registry, and fixture registry. It initially fails for unimplemented
   policies and becomes green as the following tasks land.

**Verify**

```bash
cargo test -p pymab policy --locked
uv run pytest tests/parity/test_fixture_coverage.py
```

**Commit**

```text
test: establish cross-language policy parity fixtures
```

### Task 5: Port basic action-value policies

**Create**

- `crates/pymab-core/src/policy/basic.rs`
- `crates/pymab-core/src/policy/epsilon_greedy.rs`
- `crates/pymab-core/src/policy/softmax.rs`
- `tests/fixtures/policies/random.json`
- `tests/fixtures/policies/greedy.json`
- `tests/fixtures/policies/epsilon_greedy.json`
- `tests/fixtures/policies/decaying_epsilon_greedy.json`
- `tests/fixtures/policies/softmax.json`

**Modify**

- `crates/pymab-core/src/policy/mod.rs`
- `crates/pymab-core/src/policy/registry.rs`
- `crates/pymab-core/tests/policy_fixtures.rs`

**Policies**

- `RandomPolicy`
- `GreedyPolicy`
- `EpsilonGreedyPolicy`
- `DecayingEpsilonGreedyPolicy`
- `SoftmaxPolicy`

**Tests**

- Constructor boundaries and non-finite values.
- Incremental means, epsilon schedule, stable softmax, ties, reset, clone, and
  recommendation.
- Seed reproducibility and valid stochastic support.
- Rust state-byte accounting.

**Verify**

```bash
cargo test -p pymab basic --locked
uv run pytest tests/parity/test_policy_state.py -k "random or greedy or softmax"
```

**Commit**

```text
feat: port basic policies to Rust
```

### Task 6: Port UCB-family policies

**Create**

- `crates/pymab-core/src/policy/ucb.rs`
- `tests/fixtures/policies/ucb.json`
- `tests/fixtures/policies/kl_ucb.json`
- `tests/fixtures/policies/moss.json`

**Policies**

- `UCBPolicy`
- `KLUCBPolicy`
- `MOSSPolicy`

**Tests**

- Deterministic unseen-arm order.
- Confidence bonuses and reward scaling.
- Bernoulli KL index solver convergence and tolerance.
- MOSS horizon validation and clipped log terms.
- Numerical behavior at probabilities zero and one.

**Verify**

```bash
cargo test -p pymab ucb --locked
uv run pytest tests/parity/test_policy_state.py -k "ucb or moss"
```

**Commit**

```text
feat: port UCB policies to Rust
```

### Task 7: Port gradient, Thompson, and Bayesian policies

**Create**

- `crates/pymab-core/src/policy/gradient.rs`
- `crates/pymab-core/src/policy/thompson.rs`
- `crates/pymab-core/src/policy/bayesian_ucb.rs`
- `tests/fixtures/policies/gradient.json`
- `tests/fixtures/policies/bernoulli_thompson.json`
- `tests/fixtures/policies/gaussian_thompson.json`
- `tests/fixtures/policies/bernoulli_bayesian_ucb.json`
- `tests/fixtures/policies/gaussian_bayesian_ucb.json`

**Policies**

- `GradientBanditPolicy`
- `BernoulliThompsonSamplingPolicy`
- `GaussianThompsonSamplingPolicy`
- `BernoulliBayesianUCBPolicy`
- `GaussianBayesianUCBPolicy`

**Tests**

- Gradient preference and baseline updates.
- Beta-Bernoulli successes/failures and exact binary validation.
- Gaussian posterior mean/precision updates.
- Beta and Normal quantiles against high-precision fixture values.
- Posterior moments, recommendation, reset, and seeded sampling support.

**Verify**

```bash
cargo test -p pymab gradient --locked
cargo test -p pymab thompson --locked
cargo test -p pymab bayesian --locked
uv run pytest tests/parity/test_policy_state.py -k "gradient or thompson or bayesian"
```

**Commit**

```text
feat: port posterior and gradient policies to Rust
```

### Task 8: Port adversarial and pure-exploration policies

**Create**

- `crates/pymab-core/src/policy/adversarial.rs`
- `crates/pymab-core/src/policy/pure_exploration.rs`
- `tests/fixtures/policies/exp3.json`
- `tests/fixtures/policies/successive_elimination.json`
- `tests/fixtures/policies/median_elimination.json`

**Policies**

- `EXP3Policy`
- `SuccessiveEliminationPolicy`
- `MedianEliminationPolicy`

**Tests**

- EXP3 log-weight stability after long concentrated updates.
- Probability normalization and positive exploration floor.
- Active-arm masks, confidence bounds, elimination, phase transitions, and
  single-arm termination.
- Recommended arm and reset behavior.

**Verify**

```bash
cargo test -p pymab adversarial --locked
cargo test -p pymab pure_exploration --locked
uv run pytest tests/parity/test_policy_state.py -k "exp3 or elimination"
```

**Commit**

```text
feat: port adversarial and exploration policies to Rust
```

### Task 9: Port non-stationary and change-detection policies

**Create**

- `crates/pymab-core/src/policy/nonstationary.rs`
- `crates/pymab-core/src/policy/change_detection.rs`
- `tests/fixtures/policies/sliding_window_ucb.json`
- `tests/fixtures/policies/discounted_ucb.json`
- `tests/fixtures/policies/sliding_window_bernoulli_thompson.json`
- `tests/fixtures/policies/discounted_bernoulli_thompson.json`
- `tests/fixtures/policies/change_point_ucb.json`
- `tests/fixtures/policies/cusum_ucb.json`
- `tests/fixtures/policies/page_hinkley_ucb.json`

**Policies**

- `SlidingWindowUCBPolicy`
- `DiscountedUCBPolicy`
- `SlidingWindowBernoulliThompsonSamplingPolicy`
- `DiscountedBernoulliThompsonSamplingPolicy`
- `ChangePointUCBPolicy`
- `CUSUMUCBPolicy`
- `PageHinkleyUCBPolicy`

**Tests**

- Global-time window expiry and bounded deque capacity.
- Discounted counts, sums, successes, and failures.
- Detector warm-up, positive/negative drift, arm-local resets, and change counts.
- Long-trace finite-state properties.

**Verify**

```bash
cargo test -p pymab nonstationary --locked
cargo test -p pymab change_detection --locked
uv run pytest tests/parity/test_policy_state.py -k "sliding or discounted or change or cusum or hinkley"
```

**Commit**

```text
feat: port adaptive policies to Rust
```

### Task 10: Port contextual policies

**Create**

- `crates/pymab-core/src/policy/contextual.rs`
- `tests/fixtures/policies/linear_epsilon_greedy.json`
- `tests/fixtures/policies/lin_ucb.json`
- `tests/fixtures/policies/linear_thompson.json`
- `tests/fixtures/policies/logistic_contextual.json`

**Policies**

- `LinearEpsilonGreedyPolicy`
- `LinUCBPolicy`
- `LinearThompsonSamplingPolicy`
- `LogisticContextualBanditPolicy`

**Work and tests**

1. Use contiguous owned matrices and solve/factorization operations rather than
   explicit inverse calls.
2. Test context shape and finite validation, per-arm state isolation, SGD updates,
   upper-confidence values, posterior covariance, clipped sigmoid behavior, and
   recommendation.
3. Add property tests for finite scores, symmetric state matrices, and valid
   actions over generated well-conditioned contexts.
4. Document any tolerance wider than `1e-12` with the fixture and reason.

**Verify**

```bash
cargo test -p pymab contextual --locked
uv run pytest tests/parity/test_policy_state.py -k "linear or lin_ucb or logistic"
uv run pytest tests/parity/test_fixture_coverage.py
```

**Commit**

```text
feat: port contextual policies to Rust
```

## Phase 3: Bind policies and preserve the Python API

### Task 11: Preserve a private Python reference policy backend

**Create**

- `src/pymab/_reference/__init__.py`
- `src/pymab/_reference/policies/__init__.py`
- `src/pymab/_reference/policies/*.py` corresponding to every current policy
- `src/pymab/_reference/registry.py`
- `tests/test_reference_backend.py`

**Modify**

- Existing `src/pymab/policies/*.py`
- `src/pymab/policies/__init__.py`

**Work**

1. Move the current Python implementations into the private reference namespace
   without changing their formulae.
2. Keep `Policy`, `ContextualPolicy`, and `ActionValuePolicy` usable by custom
   Python subclasses.
3. Add a registry that constructs a fresh reference policy from a public built-in
   wrapper's immutable configuration.
4. Prove the move is behavior-neutral by running the existing policy suite against
   the reference registry before enabling native wrappers.

**Verify**

```bash
uv run pytest tests/test_policies.py tests/test_policy_contracts.py tests/test_contextual.py
uv run pytest tests/test_reference_backend.py
```

**Commit**

```text
refactor: isolate Python reference policies
```

### Task 12: Add the generic PyO3 policy handle

**Create**

- `crates/pymab-python/src/error.rs`
- `crates/pymab-python/src/policy.rs`
- `src/pymab/policies/_native_mixin.py`
- `tests/test_native_policy_api.py`
- `tests/parity/test_native_policy_parity.py`

**Modify**

- `crates/pymab-python/src/lib.rs`
- `src/pymab/policies/*.py`
- `src/pymab/policies/policy.py`

**Work**

1. Expose one opaque `_NativePolicy` PyClass backed by the Rust built-in enum.
   Constructors use typed per-policy factory functions, not unchecked arbitrary
   state dictionaries.
2. Expose select, contextual select, update, contextual update, recommend, reset,
   clone, immutable configuration, state snapshot, derived diagnostics, and
   estimated state bytes.
3. Map Rust errors to existing Python exception categories and preserve useful
   field/range/shape text.
4. Implement a NumPy RNG adapter for direct Python method calls by drawing seed
   bytes from the supplied `np.random.Generator` and running one native selection.
   Direct calls remain reproducible but are not bitwise compatible with NumPy's
   previous sampling distributions.
5. Convert every public built-in policy class into a thin native wrapper while
   preserving names, constructor parameters, methods, properties, capabilities,
   clone/reset behavior, and useful `repr` output.
6. Return copies or read-only arrays for state. Update tests that mutated public
   state to assert clone independence without relying on mutation of native state.

**Verify**

```bash
uv run maturin develop --manifest-path crates/pymab-python/Cargo.toml
uv run pytest tests/test_native_policy_api.py tests/parity/test_native_policy_parity.py
uv run pytest tests/test_policies.py tests/test_policy_contracts.py tests/test_contextual.py
cargo test --workspace --locked
```

**Commit**

```text
feat: back Python policies with Rust state
```

## Phase 4: Port environments and the experiment hot loop

### Task 13: Port rewards, priors, dynamics, and environments

**Create**

- `crates/pymab-core/src/distribution.rs`
- `crates/pymab-core/src/environment/mod.rs`
- `crates/pymab-core/src/environment/classic.rs`
- `crates/pymab-core/src/environment/contextual.rs`
- `crates/pymab-core/src/environment/dynamics.rs`
- `crates/pymab-core/tests/environment_contract.rs`
- `tests/parity/test_environment_parity.py`

**Modify**

- `src/pymab/distributions.py`
- `src/pymab/environments.py`

**Port**

- Gaussian, Bernoulli, and Uniform reward models.
- Gaussian, Beta, and Uniform arm priors.
- Stationary, gradual drift, abrupt shift, probability drift, and random arm swap
  dynamics.
- Classic bandit, linear contextual, and logistic contextual environments.

**Add**

- `FixedContextProvider` for deterministic native contextual experiments.
- `GaussianContextProvider` for stochastic native contextual experiments.

Callable context providers and custom reward/dynamics objects remain Python-only
and force fallback in `auto` mode.

**Verify**

```bash
cargo test -p pymab environment --locked
uv run pytest tests/parity/test_environment_parity.py tests/test_core.py tests/test_contextual.py
```

**Commit**

```text
feat: port environments and rewards to Rust
```

### Task 14: Implement the Rust experiment runner and result buffers

**Create**

- `crates/pymab-core/src/experiment.rs`
- `crates/pymab-core/src/result.rs`
- `crates/pymab-core/tests/experiment_contract.rs`
- `crates/pymab-python/src/experiment.rs`
- `src/pymab/_backend.py`
- `tests/parity/test_experiment_parity.py`

**Modify**

- `crates/pymab-python/src/lib.rs`
- `src/pymab/simulation.py`
- `src/pymab/_experiment.py`
- `src/pymab/_experiment_storage.py`
- `src/pymab/provenance.py`
- `src/pymab/results.py`
- `src/pymab/types.py`
- `src/pymab/__init__.py`

**Work**

1. Implement preallocated contiguous result buffers with the current public axis
   order and dtypes.
2. Port the full replicate/step/policy loop, common versus independent reward
   coupling, recommendations, expected rewards, optimal mask, arm means, optional
   contexts, and context digest.
3. Release the GIL around native execution and transfer each completed buffer to
   NumPy once.
4. Add `backend="auto" | "rust" | "python"` validation to `ExperimentConfig`.
5. Build a compatibility report listing every component that prevents native
   execution. `rust` raises one aggregated `CompatibilityError`; `auto` selects
   the reference runner; `python` always selects it.
6. Record actual backend and RNG scheme in provenance and persisted results.
7. Keep `_ExperimentRunner` as the reference implementation and rename private
   modules only where doing so improves clarity without breaking imports.

**Parity checks**

- Shapes, dtypes, finite values, reward domains, and recommendation ranges.
- Backend-independent deterministic policy traces.
- Independent reproducibility within each backend.
- Policy-order and added-policy stream isolation in Rust.
- Reward coupling and context-digest invariants.
- Explicit fallback for callable contexts and custom policies.

**Verify**

```bash
cargo test -p pymab experiment --locked
uv run maturin develop --manifest-path crates/pymab-python/Cargo.toml --release
uv run pytest tests/parity/test_experiment_parity.py
uv run pytest tests/test_simulation_reliability.py tests/test_contextual.py
```

**Commit**

```text
feat: run built-in experiments in Rust
```

### Task 15: Complete correctness and compatibility coverage

**Create**

- `crates/pymab-core/tests/policy_properties.rs`
- `crates/pymab-core/tests/no_panics.rs`
- `crates/pymab-core/README.md`
- `tests/parity/test_error_parity.py`
- `tests/parity/test_reproducibility.py`

**Modify**

- Existing Python policy, environment, result, persistence, and reliability tests.
- `crates/pymab-core/src/lib.rs`
- `docs/source/reliability.rst`
- `docs/source/policy_assumptions.rst`

**Work**

1. Add `proptest` invariants for all policy families and environments.
2. Fuzz malformed public inputs through safe constructors and update methods and
   assert errors rather than panics.
3. Finish the 27-policy fixture registry gate.
4. Run the entire existing Python suite with native wrappers installed.
5. Run reference-only test subsets to ensure fallback has not drifted.
6. Confirm saved results from before the migration still load.
7. Add compiling Rust examples for a policy loop, a complete experiment, a custom
   policy trait implementation, typed error handling, and reproducible streams.
   Use the crate-specific README as the crates.io landing page.

**Verify**

```bash
cargo test --workspace --all-features --locked
cargo test --doc --workspace --locked
uv run pytest --cov=pymab --cov-branch --cov-fail-under=92
make docs
```

**Commit**

```text
test: prove Rust Python behavioral parity
```

## Phase 5: Produce performance and memory evidence

### Task 16: Add Rust microbenchmarks and allocation accounting

**Create**

- `crates/pymab-core/benches/policies.rs`
- `crates/pymab-core/benches/experiments.rs`
- `crates/pymab-core/src/memory.rs`
- `crates/pymab-core/tests/memory_contract.rs`

**Modify**

- `crates/pymab-core/Cargo.toml`
- Each Rust policy state type.

**Work**

1. Add Criterion groups covering select, update, select-plus-update, classic
   experiments, and contextual experiments.
2. Prevent optimizer elimination with black-boxed inputs and outputs.
3. Implement capacity-aware state-size reporting for vectors, deques, masks, and
   matrices, with tests for lower bounds and growth behavior.
4. Record policy configuration and state size beside each benchmark result.

**Verify**

```bash
cargo bench -p pymab --no-run --locked
cargo test -p pymab memory_contract --locked
```

**Commit**

```text
perf: benchmark Rust policies and experiments
```

### Task 17: Add cross-backend time and memory harness

**Create**

- `benchmarks/worker.py`
- `benchmarks/run_backends.py`
- `benchmarks/memory.py`
- `benchmarks/report.py`
- `benchmarks/thresholds.toml`
- `tests/test_benchmark_harness.py`
- `docs/source/performance.rst`

**Modify**

- `benchmarks/cases.py`
- `docs/source/index.rst`
- `pyproject.toml`
- `uv.lock`
- `Makefile`

**Work**

1. Run each backend in a separate child process with identical workload config.
2. Use `psutil` from the parent to sample child RSS, capturing post-import
   baseline, peak, and incremental peak. Keep workloads long enough that sampling
   resolution cannot miss the steady-state peak.
3. Record median elapsed time, decisions per second, output bytes, per-policy
   state bytes, and complete environment metadata in JSON.
4. Generate `performance.rst` from JSON; no benchmark numbers are typed by hand.
5. Enforce the approved criteria: every case faster in Rust, aggregate classic
   and contextual suites at least 2x faster, every native policy state smaller,
   and aggregate incremental peak RSS lower.
6. Run the full harness locally in release mode and commit the raw result plus
   generated report from the documented machine.

**Verify**

```bash
uv run maturin develop --manifest-path crates/pymab-python/Cargo.toml --release
uv run pytest tests/test_benchmark_harness.py
uv run python -m benchmarks.run_backends --all --output benchmarks/results/local.json
uv run python -m benchmarks.report benchmarks/results/local.json --check-thresholds
make docs
```

**Commit**

```text
perf: demonstrate native speed and memory gains
```

## Phase 6: CI, packaging, and dual publication

### Task 18: Extend pull-request CI for Rust and native wheels

**Create**

- `.github/workflows/performance.yml`
- `scripts/check_policy_coverage.py`
- `scripts/check_versions.py`

**Modify**

- `.github/workflows/ci.yml`
- `Makefile`
- `.github/SECURITY.md`

**Work**

1. Add pinned Rust setup, Cargo cache, fmt, Clippy-with-denied-warnings, workspace
   tests, docs, MSRV 1.83, and Rust dependency audit jobs.
2. Build the extension before Python test jobs on 3.11-3.14.
3. Add parity and policy-registry coverage gates.
4. Add `cargo package` and `cargo publish --dry-run` for the public crate.
5. Build representative Linux, macOS, and Windows wheels in PR CI and run an
   isolated native smoke test.
6. Make the performance workflow scheduled and manually dispatchable. PR CI only
   compiles and smoke-runs benchmarks; the fixed Linux performance job enforces
   relative thresholds and uploads raw JSON plus rendered documentation.

**Verify**

```bash
make ci
cargo package -p pymab --locked
cargo publish -p pymab --dry-run --locked
uv run python scripts/check_policy_coverage.py
uv run python scripts/check_versions.py
```

**Commit**

```text
ci: validate Rust core and native wheels
```

### Task 19: Replace the Python-only release with verified dual publication

**Modify**

- `.github/workflows/release.yml`
- `.github/workflows/release-please.yml`
- `release-please-config.json`
- `.release-please-manifest.json`
- `pyproject.toml`
- `docs/RELEASING.md`
- `README.md`

**Create**

- `scripts/check_registry_version.py`
- `scripts/verify_release_artifacts.py`
- `docs/source/migration_rust.rst`

**Work**

1. Configure Release Please to update the Cargo workspace version, lock file,
   changelog, and manifest without embedding a version in workflow code.
2. Build the `.crate`, one Python sdist, and the complete CPython 3.11-3.14 wheel
   matrix for Linux x86-64/aarch64, macOS x86-64/arm64, and Windows x86-64 using
   pinned Maturin actions and explicit Linux compatibility tags.
3. Test every wheel on its target platform and verify version, import, native
   backend identity, and a native experiment.
4. Verify the `.crate` contents and build it from the packaged archive.
5. Make all publication jobs depend on all artifact verification jobs.
6. Publish `pymab` to crates.io first and Python artifacts to PyPI second. Query
   both registries and skip only when the exact target version already exists so
   a partial-release retry is safe.
7. Use the existing PyPI trusted publisher. Document the one-time crates.io first
   release with a scoped token, then configure `rust-lang/crates-io-auth-action`
   for OIDC trusted publishing on later releases.
8. Attach verified artifacts and the benchmark report to the GitHub release.
9. Document release rollback as yanking, never overwriting or deleting immutable
   registry artifacts.

**Verify**

```bash
cargo publish -p pymab --dry-run --locked
uv run maturin build --release --sdist --manifest-path crates/pymab-python/Cargo.toml
uv run python scripts/verify_release_artifacts.py --dist dist --cargo-package target/package
uv run python scripts/check_versions.py
make docs
```

**Commit**

```text
ci: publish matching Rust and Python releases
```

## Phase 7: Completion audit

### Task 20: Audit every requirement and artifact

**Create**

- `docs/rust-core-completion-audit.md`

**Audit matrix**

1. Compare the 27 exported Python policy names with Rust registry entries,
   Python wrappers, reference constructors, fixtures, Rust tests, and parity
   tests. Every row must have direct evidence.
2. Verify every reward model, prior, dynamic, and built-in environment has the
   intended native implementation or documented fallback classification.
3. Run all Rust and Python correctness, parity, documentation, package, security,
   and benchmark gates from a clean checkout.
4. Inspect the built `.crate`, sdist, and wheels rather than relying only on build
   commands.
5. Install artifacts outside the source tree and run native and fallback smoke
   tests.
6. Verify the benchmark report meets every approved time and memory criterion.
7. Inspect CI dependency graphs to prove neither registry is contacted before all
   artifacts pass.
8. Verify registry authentication and first-release instructions are complete,
   while leaving the eventual version unset until release preparation.

**Full local gates**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo test --doc --workspace --locked
cargo package -p pymab --locked
cargo publish -p pymab --dry-run --locked
make format
make lint
make security
make test
make docs
uv run pytest tests/parity
uv run python -m benchmarks.run_backends --all --output benchmarks/results/final.json
uv run python -m benchmarks.report benchmarks/results/final.json --check-thresholds
uv run python scripts/verify_release_artifacts.py --dist dist --cargo-package target/package
```

**Commit**

```text
docs: record Rust core completion evidence
```

The migration is complete only when the audit contains authoritative evidence for
every objective and all commands above pass. A partial policy port, a benchmark
without same-machine comparison, or workflows that have not built and inspected
their artifacts do not satisfy the objective.
