# Changelog

## Unreleased

### Added

- PyMAB Arcade, a static browser experience for learning epsilon-greedy and
  LinUCB with the real package wheel, an inspectable decision view, and a
  disposable Python Lab.
- A shared typed statistical core with deterministic, memory-bounded event,
  cluster, ratio, paired-difference, and replicate-curve bootstrapping.
- Immutable benchmark summary and comparison records with explicit JSON and
  pandas conversion boundaries.

### Changed

- Benchmarking, offline estimation, and plotting now share
  `BootstrapConfig`; separate bootstrap keyword arguments and
  `BootstrapBandConfig` were removed.
- Simulation execution, result validation, and persistence schemas now use
  focused internal modules behind their public facades. NPZ writes build
  metadata without converting result tensors to Python lists.
- The repository branch-coverage gate increased from 90% to 92%.

### Removed

- `SimulationResult` is no longer re-exported from `pymab.simulation`. Import it
  from `pymab.results` or the package root.

## [2.0.1](https://github.com/danielaLopes/pymab/compare/v2.0.0...v2.0.1) (2026-09-15)


### Bug Fixes

* complete release publication flow ([1dbd72b](https://github.com/danielaLopes/pymab/commit/1dbd72bac1c9412e4d738c53d574031f18924472))
* fail closed on release tag lookup ([11aaf69](https://github.com/danielaLopes/pymab/commit/11aaf6969551f8e4ea50ddeeda0f5ca957e8ba41))

## [2.0.0](https://github.com/danielaLopes/pymab/compare/v2.0.0...v2.0.0) (2026-09-15)


### ⚠ BREAKING CHANGES

* separate simulation result persistence boundaries
* consolidate statistical analysis APIs
* harden v2 reliability and architecture

### Features

* add all PyMAB policies to Arcade ([f241034](https://github.com/danielaLopes/pymab/commit/f24103446ad82022397ede46566256df0970bc1c))
* add applied contextual scenarios ([00104bc](https://github.com/danielaLopes/pymab/commit/00104bc9753e069f1fd22f335b80fbda6c3bbd79))
* add benchmarking and real-world examples ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* add benchmarking and real-world examples ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* add contextual, adversarial, non-stationary, and pure-exploration policies ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* add contextual, adversarial, non-stationary, and pure-exploration policies ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* add deterministic demo lesson bridge ([ff91445](https://github.com/danielaLopes/pymab/commit/ff9144525e9e9a5de6848418ca5e9ff8d1b9642a))
* add Rust core contracts and deterministic streams ([4672ee6](https://github.com/danielaLopes/pymab/commit/4672ee66115f1490c869ea9b41d109d730df2d61))
* add universal decision history board ([8f5efd4](https://github.com/danielaLopes/pymab/commit/8f5efd4f5210a03228148a7f05517fc00fe12d42))
* add validated snapshot code viewer ([7c33a36](https://github.com/danielaLopes/pymab/commit/7c33a36ddea2317972d274774644ad056f0e70df))
* automate releases with Release Please ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* automate releases with Release Please ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* back Python policies with Rust state ([78e7c22](https://github.com/danielaLopes/pymab/commit/78e7c226f647a130170e7260d63ec43adf0c7f64))
* build pymab arcade web experience ([1807571](https://github.com/danielaLopes/pymab/commit/18075719cc5ca8c4fdb0c0267ec2693ced797260))
* consolidate statistical analysis APIs ([72f362f](https://github.com/danielaLopes/pymab/commit/72f362f14aa80b251235378566d359d4ab0bf663))
* customize recommendation candidates ([4d5adab](https://github.com/danielaLopes/pymab/commit/4d5adab60cf5681f9f09c5965b6d7461a2b96b4a))
* customize recommendation context signals ([37b59b9](https://github.com/danielaLopes/pymab/commit/37b59b9d8b8d44fb685b3fb147e8aa7019d1ac81))
* deploy unified PyMAB website ([6e7d398](https://github.com/danielaLopes/pymab/commit/6e7d3985e156391ac5729bbe48cea57f3f79e192))
* explain recommendation context matrix ([a027f16](https://github.com/danielaLopes/pymab/commit/a027f16c9efed6ffc5bb3f3ca279cbae20d534ff))
* harden v2 reliability and architecture ([83eaff8](https://github.com/danielaLopes/pymab/commit/83eaff8dc8a1ca906672cb4ac15f9a4e5ab7b5bd))
* launch the PyMAB Arcade and applied scenarios ([24ac8df](https://github.com/danielaLopes/pymab/commit/24ac8dfe59735f58efc1531e3b0d7e71db0f2b5c))
* polish guided lesson explanations ([bd4f49f](https://github.com/danielaLopes/pymab/commit/bd4f49f7300cd54ea13ccff25257270d3097864a))
* port adaptive policies to Rust ([9467e5a](https://github.com/danielaLopes/pymab/commit/9467e5aa6b0af19f6e8a7d9220319741794a84ac))
* port adversarial and exploration policies to Rust ([d066d60](https://github.com/danielaLopes/pymab/commit/d066d60234283f32fde7c5237fec49f9414bb6e9))
* port basic policies to Rust ([aa082e6](https://github.com/danielaLopes/pymab/commit/aa082e67a89d649a060ef98f2c06b3182d13118f))
* port contextual policies to Rust ([6932195](https://github.com/danielaLopes/pymab/commit/6932195a2b17f86ee294ebca6a568704eae8ca8b))
* port environments and rewards to Rust ([89735ba](https://github.com/danielaLopes/pymab/commit/89735baea36c30b73d2a54a93852b36754f9a556))
* port posterior and gradient policies to Rust ([fdc19e1](https://github.com/danielaLopes/pymab/commit/fdc19e1a3e4a40b0a7f2cf233f717ea70e1ae1cc))
* port UCB policies to Rust ([f03d91c](https://github.com/danielaLopes/pymab/commit/f03d91c3b608e11b38c77745d01b6314d9e2bf50))
* prioritize scenario decision history ([68a2dd5](https://github.com/danielaLopes/pymab/commit/68a2dd55337d7e5300050ba4e44b1ecca143d1e3))
* refresh PyMAB branding ([98cf42c](https://github.com/danielaLopes/pymab/commit/98cf42ce5f1b886ac2231a8ca8f6627c48a993bf))
* render scenario summary as terminal line ([a92670d](https://github.com/danielaLopes/pymab/commit/a92670d3065b3d454d8c409d1751955f7aa5da87))
* revamp simulation APIs and policy implementations ([b4a3dec](https://github.com/danielaLopes/pymab/commit/b4a3dec596e611878a2a1c7f3a6f4b45d387e069))
* revamp simulation APIs and policy implementations ([28e0c53](https://github.com/danielaLopes/pymab/commit/28e0c53d764ec397e3c1f27d2107caa0f204183e))
* run built-in experiments in Rust ([f888448](https://github.com/danielaLopes/pymab/commit/f888448e4e9ab8d771f974d3e221424e4e835882))


### Bug Fixes

* bootstrap unreleased workspace version ([aec803d](https://github.com/danielaLopes/pymab/commit/aec803d462fd858a2a72f55b80d56003ab149499))
* honor v2 release override ([a08eaef](https://github.com/danielaLopes/pymab/commit/a08eaef66ed9a2802ae62597aa21896d4a71d80a))
* install benchmark dependency in minimum CI ([a603341](https://github.com/danielaLopes/pymab/commit/a6033416c09f8420d29e9d70cad140c878b62939))
* make policy switching immediate ([e65fefb](https://github.com/danielaLopes/pymab/commit/e65fefb815864e1dacf770e613f7801b879c6480))
* parse annotated workspace version ([8a3b28e](https://github.com/danielaLopes/pymab/commit/8a3b28efa07f145a37e8200a9d90c55205d5f729))
* prevent bootstrap release downgrade ([b577f50](https://github.com/danielaLopes/pymab/commit/b577f50e6ecd6ebe0171ae599c8377ed74b27355))
* stabilize ci validation ([a1b6f55](https://github.com/danielaLopes/pymab/commit/a1b6f557f1544c753b7532ada62c7518ec405864))
* support inherited workspace release version ([f36e340](https://github.com/danielaLopes/pymab/commit/f36e34087a56cf1f016973d982bedf666c8bb87e))
* support inherited workspace release version ([d645a85](https://github.com/danielaLopes/pymab/commit/d645a8540b5e2570f0ffd8fe6de0e945ffcf750f))
* track shared web utility ([839df3f](https://github.com/danielaLopes/pymab/commit/839df3f6f10070f9692e44a2ddc65f1fc1c7b4f9))
* upgrade vulnerable cryptography dependency ([bcad926](https://github.com/danielaLopes/pymab/commit/bcad926f3697134fd5c6da8d6f07630975540a54))
* upgrade vulnerable cryptography dependency ([8a63d48](https://github.com/danielaLopes/pymab/commit/8a63d487629096b6a018a0226024991bd8f2959d))
* use CVSS 4-compatible Rust audit ([2b7f11d](https://github.com/danielaLopes/pymab/commit/2b7f11ddf94961b399ba5a17145a2ac3797f73c5))
* use scenario action symbols in decision state ([64af347](https://github.com/danielaLopes/pymab/commit/64af3477a805a4f0114710e20dca32d551b1aca1))


### Performance Improvements

* demonstrate native speed and memory gains ([755302f](https://github.com/danielaLopes/pymab/commit/755302f02e7e2812df3bc64309e97fa7d4ae541e))


### Code Refactoring

* separate simulation result persistence boundaries ([fe7484a](https://github.com/danielaLopes/pymab/commit/fe7484a72eeb9f2328217c97d143cd9c9f788aba))

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
