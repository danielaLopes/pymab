# Changelog

## Unreleased

### Added

- A shared typed statistical core with deterministic, memory-bounded event,
  cluster, ratio, paired-difference, and replicate-curve bootstrapping.
- Immutable benchmark summary and comparison records with explicit JSON and
  pandas conversion boundaries.

### Changed

- Benchmarking, offline estimation, and plotting now share
  `BootstrapConfig`; separate bootstrap keyword arguments and
  `BootstrapBandConfig` were removed.
- The repository branch-coverage gate increased from 90% to 92%.

## [2.0.0] - 2026-08-09

### Added

- Deterministic experiments with named random-number streams, stable policy IDs,
  multiple replicates, and common or independent reward coupling.
- Immutable, schema-versioned simulation results with JSON and NPZ persistence.
- Atomic persistence, schema migrations, recursively immutable metadata, and
  automatic Python/NumPy/component provenance with optional context recording.
- Paired bootstrap comparisons, standard errors, baseline deltas, and
  recommendation metrics for best-arm identification.
- Logged-bandit datasets, IPS, SNIPS, doubly robust estimation, overlap
  diagnostics, and sequential replay evaluation.
- Raw and clipped importance-weight diagnostics, cluster bootstrap, vectorized
  target policies, explicit zero-overlap failures, and propensity-aware replay.
- Explicit policy capabilities and separate reward-model and arm-prior APIs.
- Linear and logistic contextual environments, bounded probability drift, and
  a migration guide for the intentionally breaking v2 API.

### Changed

- Moved the package to a `src/` layout and made NumPy the only required runtime
  dependency. Pandas, plotting, SciPy, and documentation support are optional.
- Policy constructors now use one explicit vocabulary (`n_arms`, `n_features`)
  and policies implement an explicit clone/reset contract.
- Sliding-window policies now expire observations by global decision time;
  EXP3 uses stable log weights; UCB-style bounds expose reward scale.
- Offline evaluation, result persistence, provenance, non-stationary policies,
  and change detection now have focused modules behind curated facades.
- Optimal-action calculations treat statistically indistinguishable arms as
  ties using a documented numerical tolerance.
- Documentation CI now performs clean strict builds on Python 3.11 and 3.14,
  executes doctests and README snippets, enforces API coverage, checks external
  links separately, and publishes rendered HTML artifacts.

### Removed

- The v1 `Game` facade, compatibility aliases, environment mixins, implicit
  output directories, and import-time logging/plot configuration.

### Fixed

- Shared-RNG coupling that made a policy's result depend on policy order or on
  unrelated policies included in an experiment.
- Bernoulli drift leaving the valid probability domain and matrix inversion in
  contextual policies.
- Lossy integer coercion, shared mutable extension state, broken NumPy dataclass
  equality, misleading zero-overlap intervals, and unbounded plot bootstrap
  allocations.

## [1.0.0](https://github.com/danielaLopes/pymab/compare/v0.1.0...v1.0.0) (2026-08-09)


### Features

* add benchmarking and real-world examples ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* add benchmarking and real-world examples ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* add contextual, adversarial, non-stationary, and pure-exploration policies ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* add contextual, adversarial, non-stationary, and pure-exploration policies ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* automate releases with Release Please ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* automate releases with Release Please ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* revamp simulation APIs and policy implementations ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* revamp simulation APIs and policy implementations ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))


### Bug Fixes

* upgrade vulnerable cryptography dependency ([bcad926](https://github.com/danielaLopes/pymab/commit/bcad926f3697134fd5c6da8d6f07630975540a54))
* upgrade vulnerable cryptography dependency ([8a63d48](https://github.com/danielaLopes/pymab/commit/8a63d487629096b6a018a0226024991bd8f2959d))

0.1.0 - 2024-10-31
Added

Initial release of PyMAB
Implementation of basic Multi-Armed Bandit algorithms:

Epsilon-Greedy Policy
Greedy Policy
UCB Policy
Bayesian UCB Policy
Thompson Sampling Policy


Support for different reward distributions:

Gaussian
Bernoulli
Uniform


Support for different environments:
Stationary
Gradual Change
Abrupt Change
Random Arm Swapping


Game class for running bandit simulations
Visualization tools for reward distributions and performance metrics

Changed
None
Deprecated
None
Removed
None
Fixed
None
Security
None
