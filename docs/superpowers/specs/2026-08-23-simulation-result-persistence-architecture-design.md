# PyMAB Simulation, Result, and Persistence Architecture Design

Date: 2026-08-23
Status: Approved for implementation
Target branch: `architecture-cleanup`

## Purpose

Separate experiment orchestration, runtime execution, result validation, and
result persistence into focused modules with one-way dependencies. Preserve the
behavioral guarantees established by the reliability and statistical-core work
while making each unit smaller, independently testable, and easier to extend.

This branch is intentionally limited to simulation, result, and persistence
boundaries. Environment implementations and policy algorithms are out of scope.
The v2 API may change where a clearer canonical import or stricter contract
improves maintainability.

## Problems to solve

The current modules combine responsibilities that change for different reasons:

- `simulation.py` owns the public facade, compatibility checks, replicate and
  step execution, random-stream construction, array allocation, context hashing,
  and runtime output validation.
- `results.py` owns the public result, derived metrics, serialization delegates,
  pandas conversion, array normalization, and cross-field validation.
- `persistence.py` owns JSON and NPZ codecs, atomic filesystem operations,
  payload migration, schema validation, provenance reconstruction, and result
  construction.
- `SimulationResult` is re-exported from `pymab.simulation`, obscuring its
  canonical domain module.
- NPZ saving calls `SimulationResult.to_dict()` and then removes array fields,
  transiently converting every large NumPy array into Python lists even though
  an NPZ archive stores those arrays directly.
- Experiment execution imports the package root at runtime to obtain the
  version, creating an avoidable dependency back edge.

## Alternatives considered

### A. Refactor only inside the existing modules

This minimizes file movement, but the large modules would continue to contain
multiple ownership boundaries. Tests would still need broad fixtures to reach
small internal behaviors.

### B. Convert each domain into a nested package

Packages such as `simulation/`, `results/`, and `persistence/` provide room for
growth, but they create deeper imports and unnecessary migration work for the
current library size.

### C. Keep public facades and extract shallow private services

This is the selected approach. Public modules remain discoverable and private
flat modules own runtime, storage, validation, and schema mechanics. It creates
clear dependency direction without introducing a framework or deep hierarchy.

## Target module structure

```text
src/pymab/
  __init__.py
  _version.py                 package-version resolution
  simulation.py               ExperimentConfig and Experiment facade
  _experiment.py              runtime runner and deterministic streams
  _experiment_storage.py      preallocated arrays and context recording
  results.py                  SimulationResult and result-facing conveniences
  _result_validation.py       immutable arrays and cross-field invariants
  persistence.py              JSON/NPZ I/O and atomic write facade
  _result_schema.py           payloads, migrations, and reconstruction
```

All private modules have an empty `__all__`. Public modules explicitly list
their supported exports.

## Public API

`Experiment` and `ExperimentConfig` remain public from `pymab.simulation` and
the package root. `SimulationResult` remains public from `pymab.results` and the
package root, but is removed from `pymab.simulation.__all__`. Internal package
modules and tests import it from `pymab.results`, establishing one domain-owned
module path.

`ResultSerializer` remains public from `pymab.persistence`. Existing result
convenience methods (`to_dict`, `from_dict`, JSON/NPZ save/load, and
`to_pandas`) remain available.

The migration guide documents the canonical result import. No compatibility
alias is retained in `pymab.simulation` because v2 explicitly permits contract
cleanup.

## Experiment execution

`Experiment` validates the environment, policy mapping, configuration, and
capabilities at construction. `run()` creates an internal immutable run request
and delegates execution to `_ExperimentRunner`.

The runner owns replicate and step loops, cloned component graphs, named random
streams, environment state transitions, policy decisions, reward sampling, and
runtime output validation. It returns recorded arrays plus context-digest data;
the public facade captures provenance and constructs `SimulationResult`.

`_ExperimentStorage` owns all preallocation and writes. Its constructor accepts
explicit dimensions rather than importing `ExperimentConfig`, preventing a
private-to-public dependency cycle. Context hashing and optional context tensor
recording remain co-located because both consume the exact environment context
at each step.

The RNG scheme, seed derivation, policy-order invariance, reward-coupling
semantics, cloning behavior, and error messages remain unchanged.

## Result ownership and validation

`SimulationResult` owns domain data and derived views only. Its initialization
delegates normalization to `_ResultArrays` in `_result_validation.py`, which:

- strictly validates numeric, integer, and boolean arrays;
- creates read-only owned arrays;
- validates all dimensional relationships;
- validates action and recommendation bounds;
- verifies selected expected rewards against arm means;
- validates unique policy IDs and non-negative integer replicate seeds.

The public result retains immutable JSON-compatible configuration, metadata,
and provenance. Derived arrays remain computed properties and never expose
mutable internal state.

## Schema and persistence

`_result_schema.py` is the single owner of schema field sets, array-field names,
payload construction, schema migrations, field validation, provenance decoding,
and `SimulationResult` reconstruction.

It exposes private functions for:

- building metadata without array conversion;
- building a complete JSON-compatible payload;
- migrating and validating an external payload;
- constructing a validated result.

`SimulationResult.to_dict()` delegates full payload creation to this module.
`ResultSerializer.save_npz()` requests metadata-only payload creation, so it
does not allocate Python-list copies of result tensors. JSON persistence still
constructs the complete payload because JSON requires nested lists.

`persistence.py` owns path normalization, source resolution, JSON/NPZ file I/O,
and atomic writes. It wraps path-aware failures in `SerializationError`, retains
`allow_pickle=False`, flushes and fsyncs temporary files, and atomically replaces
the destination. Failed writes remove only their known sibling temporary file.

Schema version 3 and the existing schema-2 migration remain unchanged. Field
names used by JSON and NPZ are defined once.

## Version ownership

`_version.py` resolves the installed distribution version with the existing
development fallback. Both the package root and experiment provenance import
that constant. Simulation code no longer imports `pymab` during a run.

## Dependency direction

```text
__init__ -> simulation -> _experiment -> _experiment_storage
   |             |                              |
   |             +------------------------------+
   |
   +---------> results -> _result_validation
                         |
persistence -> _result_schema
      |              |
      +-----------> results
```

`results.py` uses local imports for persistence conveniences, so persistence
may reconstruct `SimulationResult` without an import-time cycle.

## Error behavior

Public validation continues to raise `ValidationError` or
`CompatibilityError`. Malformed files, unsupported schemas, invalid archive
metadata, wrong suffixes, and invalid persisted results continue to raise
`SerializationError` with the source path when available.

Internal impossible states may raise `RuntimeError`, but external input errors
must never leak as assertions, raw `KeyError`, NumPy coercion errors, or partial
filesystem results.

## Testing strategy

Tests remain behavior-focused at public boundaries and add focused private-unit
coverage where it proves architectural invariants:

1. Existing deterministic traces remain bit-for-bit stable.
2. Policy order, unrelated policies, and reward coupling remain invariant.
3. Stateful environments and policies remain isolated per replicate.
4. Runtime output errors retain policy, replicate, and step context.
5. Result arrays remain immutable and all cross-field invariants are tested.
6. JSON and NPZ round trips, corrupt inputs, suffix handling, atomic failures,
   and schema-2 migration remain covered.
7. NPZ saving succeeds when `SimulationResult.to_dict()` is instrumented to
   fail, proving metadata creation does not traverse full arrays.
8. Import smoke tests prove the canonical result path and absence of circular
   imports from both a source tree and an installed wheel.
9. Public API and migration documentation examples execute in CI.

Each new private module must reach at least 95% branch coverage. Repository
coverage must remain above the existing 92% gate.

## Completion criteria

The refactor is complete only when:

- module responsibilities and dependency direction match this design;
- no runtime import from simulation to the package root remains;
- NPZ metadata creation avoids full array-to-list conversion;
- documented public imports and persistence formats behave as specified;
- deterministic regression fixtures are unchanged;
- Ruff, formatting, strict mypy, Bandit, dependency audit, full tests, minimum
  NumPy, Python 3.11 through 3.14, strict Sphinx HTML/doctest/coverage/linkcheck,
  README snippets, notebooks/examples, wheel/sdist builds, Twine validation,
  and installed-wheel smoke/type checks pass;
- unrelated untracked plan files remain outside every commit.
