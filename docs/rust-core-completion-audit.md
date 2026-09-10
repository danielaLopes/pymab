# Rust core completion audit

**Audit date:** 2026-08-25  
**Branch:** `rust-pymab`  
**Release version:** deliberately not selected

## Outcome

The Rust core implementation is complete in the repository. All 27 public
Python policies have native Rust implementations, Python reference
implementations, shared fixtures, and cross-backend parity coverage. Built-in
classic and contextual simulations can execute through the native runner, while
unsupported or custom configurations remain available through an explicit
Python fallback.

The implementation, documentation, packages, security checks, and local
performance thresholds pass on the audit machine. Cross-platform wheel jobs and
actual registry publication are defined in CI but were not run locally and no
release was published as part of this work.

## Policy coverage

The authoritative coverage gate is `scripts/check_policy_coverage.py`. It
compares the public Python exports, `PolicyKind::ALL`, the Python reference
registry, the fixture registry, every fixture file, and constructible native
handles. The gate passed for all 27 built-ins.

| Public Python class | Rust kind | Rust implementation | Shared fixture |
| --- | --- | --- | --- |
| `BernoulliBayesianUCBPolicy` | `bernoulli_bayesian_ucb` | `bayesian_ucb.rs` | `bernoulli_bayesian_ucb.json` |
| `BernoulliThompsonSamplingPolicy` | `bernoulli_thompson_sampling` | `thompson.rs` | `bernoulli_thompson.json` |
| `ChangePointUCBPolicy` | `change_point_ucb` | `change_detection.rs` | `change_point_ucb.json` |
| `CUSUMUCBPolicy` | `cusum_ucb` | `change_detection.rs` | `cusum_ucb.json` |
| `DecayingEpsilonGreedyPolicy` | `decaying_epsilon_greedy` | `epsilon_greedy.rs` | `decaying_epsilon_greedy.json` |
| `DiscountedBernoulliThompsonSamplingPolicy` | `discounted_bernoulli_thompson_sampling` | `nonstationary.rs` | `discounted_bernoulli_thompson.json` |
| `DiscountedUCBPolicy` | `discounted_ucb` | `nonstationary.rs` | `discounted_ucb.json` |
| `EpsilonGreedyPolicy` | `epsilon_greedy` | `epsilon_greedy.rs` | `epsilon_greedy.json` |
| `EXP3Policy` | `exp3` | `adversarial.rs` | `exp3.json` |
| `GaussianBayesianUCBPolicy` | `gaussian_bayesian_ucb` | `bayesian_ucb.rs` | `gaussian_bayesian_ucb.json` |
| `GaussianThompsonSamplingPolicy` | `gaussian_thompson_sampling` | `thompson.rs` | `gaussian_thompson.json` |
| `GradientBanditPolicy` | `gradient_bandit` | `gradient.rs` | `gradient.json` |
| `GreedyPolicy` | `greedy` | `basic.rs` | `greedy.json` |
| `KLUCBPolicy` | `kl_ucb` | `ucb.rs` | `kl_ucb.json` |
| `LinUCBPolicy` | `lin_ucb` | `contextual.rs` | `lin_ucb.json` |
| `LinearEpsilonGreedyPolicy` | `linear_epsilon_greedy` | `contextual.rs` | `linear_epsilon_greedy.json` |
| `LinearThompsonSamplingPolicy` | `linear_thompson_sampling` | `contextual.rs` | `linear_thompson_sampling.json` |
| `LogisticContextualBanditPolicy` | `logistic_contextual_bandit` | `contextual.rs` | `logistic_contextual_bandit.json` |
| `MedianEliminationPolicy` | `median_elimination` | `pure_exploration.rs` | `median_elimination.json` |
| `MOSSPolicy` | `moss` | `ucb.rs` | `moss.json` |
| `PageHinkleyUCBPolicy` | `page_hinkley_ucb` | `change_detection.rs` | `page_hinkley_ucb.json` |
| `RandomPolicy` | `random` | `basic.rs` | `random.json` |
| `SlidingWindowBernoulliThompsonSamplingPolicy` | `sliding_window_bernoulli_thompson_sampling` | `nonstationary.rs` | `sliding_window_bernoulli_thompson.json` |
| `SlidingWindowUCBPolicy` | `sliding_window_ucb` | `nonstationary.rs` | `sliding_window_ucb.json` |
| `SoftmaxPolicy` | `softmax` | `softmax.rs` | `softmax.json` |
| `SuccessiveEliminationPolicy` | `successive_elimination` | `pure_exploration.rs` | `successive_elimination.json` |
| `UCBPolicy` | `ucb` | `ucb.rs` | `ucb.json` |

For every row, the remaining evidence is common:

- Rust registration: `crates/pymab-core/src/policy/registry.rs`
- Python native wrapper: `src/pymab/policies/` and
  `src/pymab/policies/_native_mixin.py`
- Python fallback constructor: `src/pymab/_reference/registry.py`
- Rust fixture execution: `crates/pymab-core/tests/policy_fixtures.rs`
- Python state and native parity:
  `tests/parity/test_policy_state.py` and
  `tests/parity/test_native_policy_parity.py`

## Native execution surface

| Surface | Native implementation | Fallback/contract evidence |
| --- | --- | --- |
| Gaussian, Bernoulli, and Uniform rewards | `crates/pymab-core/src/distribution.rs` | `crates/pymab-core/tests/environment_contract.rs` |
| Gaussian, Beta, and Uniform priors | `crates/pymab-core/src/distribution.rs` | policy fixtures and parity tests |
| Stationary, gradual, abrupt, probability, and random-swap dynamics | `crates/pymab-core/src/environment/dynamics.rs` | environment contract tests |
| Classic environments | `crates/pymab-core/src/environment/classic.rs` | Python environment tests |
| Contextual environments and contexts | `crates/pymab-core/src/environment/contextual.rs` | contextual parity tests |
| Experiment loop and result arrays | `crates/pymab-core/src/experiment.rs` | backend contract and runner parity tests |
| PyO3 boundary, GIL release, one-time NumPy transfer | `crates/pymab-python/src/experiment.rs` | native runner and concurrency tests |
| Backend selection | `src/pymab/simulation.py` | `auto`, `rust`, and `python` backend tests |
| Unsupported/custom objects | Python reference backend | compatibility report and fallback tests |

Offline evaluation, statistics, plotting, and arbitrary Python callbacks remain
Python responsibilities by design; they are not performance-critical portions
of the built-in simulation loop.

## Verification evidence

| Gate | Audit result |
| --- | --- |
| Python formatting, Ruff, and mypy | Passed via `make format lint` |
| Python suite | 390 tests passed; 93.88% coverage |
| Rust 1.83 MSRV formatting and Clippy | Passed on explicit Rust 1.83.0 with warnings denied |
| Rust 1.83 MSRV suite | 67 tests and 4 documentation tests passed |
| Policy coverage | Passed for all 27 built-ins |
| Version consistency | Python and native metadata agree; no future version was chosen |
| Documentation | Strict HTML, doctest, API coverage, and README snippet gates passed; 5 doctests and 100% API documentation coverage |
| Python security | Bandit reported zero findings; `pip-audit` reported no known vulnerabilities |
| Rust security | `cargo-audit` reported no vulnerabilities; one allowed unmaintained transitive dependency warning for `paste` |
| Crate packaging | `cargo package -p pymab --locked` and `cargo publish -p pymab --dry-run --locked` passed |
| Local artifact inspection | One native wheel, the sdist, and the `.crate` passed `scripts/verify_release_artifacts.py` |
| Reproducibility | Repeated local `cargo package` archives were byte-for-byte identical |
| Isolated install | Native wheel installed outside the source tree and completed a Rust-backend smoke run |
| Workflow syntax | All GitHub workflows passed `actionlint` 1.7.12 |
| Registry state | The workspace's current mirrored version was absent from both PyPI and crates.io at audit time; nothing was published |

The representative local wheel proves the build and isolated-import path on the
audit host. The release workflow's complete 20-wheel matrix (CPython 3.11-3.14
across supported Linux, macOS, and Windows targets) requires hosted runners and
therefore remains a pre-publication CI execution check, not a locally completed
claim.

## Performance evidence

The checked raw result is `benchmarks/results/local.json`; the generated report
is `docs/source/performance.rst`. Both backends ran in isolated processes on the
same machine, and RSS was sampled by the parent process after imports.

| Workload | Decisions | Native speedup | Python/Rust incremental peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| Stationary classic | 112,000 | 12.45x | 21.64 / 17.75 |
| Bernoulli | 16,000 | 52.02x | 62.78 / 3.78 |
| Non-stationary | 80,000 | 20.47x | 18.50 / 14.66 |
| Contextual | 8,000 | 23.65x | 4.38 / 1.47 |

The aggregate classic speedup is 21.94x, the contextual speedup is 23.65x, and
the aggregate Rust/Python incremental RSS ratio is 0.351. Capacity-aware native
policy state was smaller for every shared trace; the worst Rust/Python state
ratio among the 27 policies was 0.159.

## Publication safety

`.github/workflows/release.yml` encodes this dependency chain:

```text
metadata -> crate + sdist + 20 wheels -> verify -> registry-state
         -> publish-crate -> publish-pypi -> attach
```

No registry write can start until the crate, sdist, every wheel, target-native
wheel smoke tests, version checks, and aggregate artifact verification pass.
Publication is idempotent for an already-present exact version. crates.io is
published first so a crate failure cannot leave a PyPI-only release.

Trusted-publishing setup and the one-time, scoped crates.io bootstrap-token path
are documented in `docs/RELEASING.md`. Release Please reads the Cargo manifest;
the eventual release version remains unset until the normal release process
chooses it.

## Remaining external validation

Before a real release, push `rust-pymab`, let the hosted CI and full wheel matrix
complete, and perform the documented trusted-publisher setup. Those actions were
not authorized or necessary for this implementation audit. The repository is
otherwise ready for review without creating a separate repository.
