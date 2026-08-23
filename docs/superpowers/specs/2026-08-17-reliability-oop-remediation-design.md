# PyMAB Reliability and Object-Oriented Remediation Design

Date: 2026-08-17
Status: Implemented and verified
Target branch: `improve-reliability`

## Purpose

This design closes the remaining correctness, statistical-validity, extension,
performance, packaging, and maintainability gaps identified in the v2 review.
Because v2 is already a breaking release, the implementation prioritizes clear,
strict, maintainable contracts over compatibility with the intermediate v2
worktree. Convenient facades remain where they improve usability, but obsolete
or ambiguous contracts are removed instead of retained through compatibility
layers.

The work is complete only when every behavior described here has regression
coverage and the complete CI-equivalent suite passes against both the source
tree and a built wheel.

## Design principles

1. Model domain concepts as focused immutable value objects where practical.
2. Use composition and protocols for replaceable behavior; use inheritance only
   for the existing policy algorithm hierarchy where it remains cohesive.
3. Validate external data before coercing it.
4. Keep statistical assumptions explicit in types, results, and documentation.
5. Expose one deliberate public import path per concept and document migrations.
6. Keep randomness injectable and reproducible; never create hidden entropy in
   a public sampling API.
7. Prefer explicit failures to silent recovery from corrupt numerical state.

## Alternatives considered

### A. Patch each defect in the existing modules

This minimizes file movement but leaves validation, persistence, statistical
estimation, and orchestration coupled in large functions. It would address the
symptoms without producing clear ownership boundaries.

### B. Replace the package with a new framework

This could produce idealized interfaces but would create unnecessary migration
work and risk regressions in the already improved v2 API.

### C. Domain refactor with a deliberate v2 public API

This is the selected approach. Focused domain objects and services are extracted
where responsibilities are already distinct. Public exports are intentionally
curated after the refactor rather than preserving every intermediate path.
Behavioral and contract changes are allowed when they improve correctness,
clarity, or long-term maintainability, and the migration guide records them.

## Target architecture

The package will remain shallow. New modules are added only for genuine domains:

```text
src/pymab/
  __init__.py
  errors.py                 package exception hierarchy
  validation.py             strict reusable boundary validators
  provenance.py             immutable JSON values and run provenance
  results.py                SimulationResult and schema validation
  persistence.py            atomic JSON/NPZ persistence
  _random.py                internal deterministic stream derivation
  simulation.py             ExperimentConfig and Experiment orchestration facade
  benchmarking.py           paired replicate analysis
  offline/
    __init__.py             stable offline facade
    data.py                 LoggedBanditDataset and result records
    estimators.py           IPS, SNIPS, and DR
    replay.py               sequential replay
    bootstrap.py            event, cluster, and replicate resampling
  policies/
    ...                     existing stable policy facade
    ucb.py                   stationary UCB and MOSS
    nonstationary.py         windowed and discounted policies
    change_detection.py     change-point UCB variants
```

The existing `offline.py` module will become the `offline` package shown above.
Its `__init__.py` defines the supported offline API. Internal implementation
types are not re-exported merely for compatibility.

## Domain errors

`pymab.errors` will define:

- `PyMABError`: package base exception.
- `ValidationError`: invalid external data or configuration; subclasses
  `ValueError` for compatibility.
- `CompatibilityError`: incompatible environment/policy capabilities;
  subclasses `TypeError` for compatibility.
- `OverlapError`: an unidentifiable offline estimate; subclasses
  `ValidationError`.
- `SerializationError`: invalid or corrupt result persistence.

Compatibility inheritance from built-in exceptions is used only when it makes
the exception semantically correct, not solely to preserve old catch clauses.

## Strict validation

Reusable validators will accept array-like values but inspect the original
values before conversion. Integer-labelled inputs must reject booleans,
strings, nonfinite values, fractional numbers, and out-of-range integers.

The same validator will be used for:

- logged actions;
- simulation actions and recommendations;
- replay-selected actions;
- replicate seeds and integer configuration values.

Probability and floating-point validators will reject nonnumeric inputs and
nonfinite values with field-specific messages. Persisted payloads will be
validated as a schema before object construction rather than cast field by
field. Policy identifiers must already be nonempty strings.

## Equality and immutable data

Stateful policies and mutable environments will use `eq=False`; identity is the
only general equality semantics for mutable algorithms.

Array-backed immutable records will also disable generated dataclass equality.
Where value comparison is useful, they will expose an explicit `equals()`
method using `numpy.array_equal` plus scalar and metadata comparisons.

Both result configuration and metadata will use one recursively immutable,
JSON-compatible value model. Nonfinite numbers and unsupported objects are
rejected. No nested mutable object may remain reachable through a result.

## Reproducible extension contracts

`RewardModel`, `EnvironmentDynamics`, and contextual providers will have an
explicit cloning contract. Built-in immutable implementations may safely return
themselves; mutable implementations must return independent state.

A `ContextProvider` protocol/object will replace an unconstrained callable as
the preferred interface. Plain callables remain supported through a stateless
adapter, documented as requiring referential state isolation. Stateful provider
objects implement `clone()`.

Environment cloning will clone the complete behavior graph. Regression tests
will prove that mutable custom dynamics, reward models, and context providers do
not leak state between replicates.

Environment capability protocols will describe classic and contextual
environments without requiring concrete `isinstance` checks. Experiment
orchestration will dispatch through declared capabilities and validated
protocols, allowing third-party environments to integrate safely.

## Randomness contract

All public sampling operations require an explicit `numpy.random.Generator`.
`RewardModel.sample_one()` will no longer create a hidden generator. A clearly
named convenience helper may accept a seed, but it must create and expose a
deterministic stream from that seed.

The deterministic stream derivation logic moves to one internal module. Run
provenance records the RNG scheme version and library dependency versions.

## Simulation result and persistence

`SimulationResult` remains the public result type but delegates persistence to a
dedicated serializer. The serializer will:

- normalize `.npz` and `.json` suffixes consistently for save and load;
- write to a sibling temporary file and atomically replace the destination;
- wrap malformed archives, missing fields, invalid JSON, and schema errors in
  `SerializationError` with the source path;
- retain `allow_pickle=False`;
- provide explicit schema migration functions;
- validate schema values before coercion.

Automatic provenance will include:

- PyMAB, Python, and NumPy versions;
- experiment configuration and RNG scheme version;
- serializable environment configuration;
- policy class and hyperparameter configuration by policy ID.

Contextual experiments gain an opt-in context recording mode. When disabled,
the result records a deterministic context digest when the provider makes that
possible. Large context arrays are never stored silently.

## Offline policy evaluation

### Estimator inputs

`LoggedBanditDataset` remains immutable and gains optional cluster identifiers.
Clusters represent the independent resampling unit, such as user, session, or
trajectory. Event-level bootstrap remains available only as an explicit choice.

Target probabilities may be supplied through the current event protocol or an
optional batch protocol. The implementation avoids an `N x K` allocation when
only logged-action probabilities are required by IPS or SNIPS.

### Overlap and clipping

IPS and SNIPS raise `OverlapError` when total target support on logged actions is
zero. DR may return its direct-model component under zero overlap, but the result
must report that it is model-only and unsupported by logged-action overlap.

`OfflineEstimate` records both raw and effective diagnostics:

- raw and clipped effective sample size;
- raw and clipped maximum/mean weight;
- clipping threshold and clipped fraction;
- overlap status;
- resampling unit and confidence method.

Clipping is represented in the estimator method metadata and documented as a
bias/variance trade-off. Raw overlap diagnostics are never overwritten by
clipped diagnostics.

### Uncertainty

The default remains deterministic percentile bootstrap for independent rows,
but documentation and result metadata state the independence assumption.
Supplying clusters activates cluster bootstrap. Bootstrap implementations are
chunked to a bounded memory target.

## Sequential replay

Replay will distinguish three concerns:

- `clone_policy`: evaluate an independent policy object;
- `reset_policy`: start from fresh state;
- logging design: uniform matching or propensity-aware rejection.

The default preserves current fresh-clone behavior. `clone_policy=False` will no
longer imply resetting. Invalid policy actions raise immediately with event and
policy context.

Uniform replay requires the caller to select `logging_scheme="uniform"`.
Nonuniform logging requires each logged action's propensity. After the target
policy's sampled action matches the logged action, replay applies a second
acceptance probability `c / propensity`, where `c` is no greater than the
minimum supplied propensity. This makes the combined action acceptance
probability proportional to the target policy. The default `c` is the observed
minimum propensity; callers may supply a smaller positive value. The result
includes accepted event indices, selected actions, rewards, and acceptance
diagnostics.

## Policy algorithm corrections

### Sliding-window policies

`SlidingWindowUCBPolicy` and sliding-window Bernoulli Thompson Sampling will use
global decision time. Every observation stores its decision step, and values
older than `window_size` decisions expire even when their arm is not selected.
The old pull-count-window behavior is removed rather than retained under a
second policy name.

### EXP3

EXP3 will store log weights. Probabilities will be normalized with stable
log-sum-exp arithmetic, and nonfinite state will raise a numerical error instead
of silently resetting to uniform. Parameter validation will enforce the valid
range and explain the default coupling between exploration and learning rate.

### Assumptions

Policies based on bounded/sub-Gaussian concentration will expose the scale
needed by their confidence bounds or clearly restrict their advertised reward
capabilities. MOSS validates `horizon >= n_arms`.
The policy-assumption documentation and runtime capabilities must agree.

## Benchmarking and plotting performance

Bootstrap operations will share a chunked resampler with a configurable memory
budget. Plotting will not allocate a full
`resamples x steps x policies` tensor. Plot methods expose confidence level,
resample count, and analysis seed.

`BenchmarkResult.to_dict()` computes each summary once. A private lazy cache may
be used only if it cannot expose mutable shared state; otherwise values are
computed once per operation. Public summary records gain a `TypedDict` or
immutable record type rather than unstructured `dict[str, Any]` internally.

Contextual linear policies may adopt Cholesky factorization or cached solves only
after regression and reference tests prove equivalent behavior.

## Public API and documentation

Every public module defines `__all__`. The top-level package exports the small
set of concepts needed for common workflows; specialized APIs live in cohesive
subpackages. Internal helper modules use a leading underscore unless they are
intended as stable extension surfaces. Removed or relocated v2-development
imports are documented rather than indefinitely re-exported.

Documentation will state:

- uniform/nonuniform replay requirements;
- event versus cluster bootstrap assumptions;
- raw versus clipped overlap diagnostics;
- policy boundedness, stationarity, and noise-scale assumptions;
- exact provenance guarantees and context-recording trade-offs;
- stable import paths and deprecation policy.

Sphinx coverage remains at 100%, but public docstrings must describe parameters,
returns, exceptions, and statistical assumptions rather than merely exist.

## Packaging and contributor workflow

The pip fallback will install actual development requirements rather than a
nonexistent `dev` extra. Either a mirrored `dev` extra is maintained for pip or
the fallback installs the PEP 735 group through a generated requirements file.
All optional runtime extras are installed by the documented all-features setup.

The unused Matplotlib dependency will be removed unless a supported Matplotlib
backend is implemented. The built-wheel CI job will run a minimal simulation,
offline-estimation smoke test, and consumer type check outside the source tree.

`SECURITY.md` will contain supported-version and private-reporting instructions.
Contributor documentation will list exact setup, formatting, linting, test,
documentation, package, and audit commands.

Unrelated plan files and generated artifacts will not be included in the v2
implementation commit.

## Testing strategy

Each corrected defect receives a focused regression test. Additional coverage
will include:

1. Property-based validation tests for action arrays, probability vectors,
   schemas, and persistence round trips.
2. Mutable third-party extension tests proving replicate isolation.
3. Reference or invariant tests for sliding-window algorithms, EXP3, MOSS,
   elimination policies, and posterior updates.
4. Monte Carlo checks for IPS/SNIPS/DR bias and interval coverage under known
   logging and target policies.
5. Uniform and nonuniform replay tests, including warm starts and invalid custom
   actions.
6. Corrupt, truncated, wrong-suffix, and prior-schema persistence tests.
7. Statistical confidence tests that aggregate within each replicate and treat
   replicates, not steps, as independent.
8. Performance tests with explicit peak-allocation bounds for bootstrap code.
9. Source and installed-wheel public API tests, including `py.typed`.

Randomized tests use fixed seeds and validate statistical tolerances chosen from
precomputed power/error analyses. They must not rely on one favorable trace.

## Error handling and compatibility

Corrections that reject previously coerced invalid input are intentional v2
hardening. API and return-type changes are permitted when they produce a clearer
domain model. The migration guide covers every user-visible change. When a
result schema changes, the schema version increments and a migration is
provided. No persisted payload is silently reinterpreted.

Every exception raised at an orchestration boundary includes policy ID,
replicate, step/event, and field where applicable. Low-level exceptions retain
their causes with `raise ... from ...`.

## Implementation sequence

1. Add errors, validation, immutable JSON/provenance, and regression tests.
2. Correct dataclass equality and strict action/result construction.
3. Add cloneable environment components and protocol-based environment dispatch.
4. Extract result persistence and implement atomic schema-aware I/O.
5. Refactor and correct offline estimators and replay.
6. Correct global-time window policies and stable EXP3.
7. Chunk analysis/plotting and remove duplicate summary computation.
8. Complete module extraction and facade exports.
9. Update docs, packaging, security policy, contributor workflow, and wheel CI.
10. Run all quality, documentation, statistical, package, and clean-install gates.

## Completion criteria

The implementation is complete only when:

- every reproduced defect from the review has a passing regression test;
- zero-overlap IPS cannot produce a confident numeric estimate;
- clipping cannot overwrite raw overlap diagnostics;
- replay enforces its logging design and preserves warm starts when requested;
- invalid or lossy integer coercions are rejected everywhere;
- array-backed public objects have explicit, non-broken equality semantics;
- mutable extension objects are isolated across replicates;
- sliding windows expire by global time and EXP3 remains finite under validated
  parameters;
- persistence is atomic, suffix-consistent, deeply immutable, schema-validated,
  and provenance-complete;
- bootstrap memory is bounded and benchmark summaries are not recomputed within
  one operation;
- public APIs are explicit and built-wheel smoke/type checks pass;
- Ruff, strict mypy, Bandit, dependency audit, all tests with branch coverage,
  strict Sphinx HTML/doctest/coverage/snippet/link checks, build, and strict
  Twine validation pass from a clean checkout;
- the branch contains no unrelated or generated artifacts.
