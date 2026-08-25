# Releasing PyMAB

Release Please chooses the release version and updates the single Cargo
workspace version, lock files, changelog, and release manifest. Do not choose or
hard-code a future version in the workflows. A matching release publishes the
public `pymab` Rust crate and the Python package from one tag.

## One-time repository setup

1. Configure the Release Please GitHub App credentials described in
   `.github/workflows/release-please.yml`.
2. Create protected GitHub environments named `crates-io` and `pypi`, each with
   a required reviewer.
3. Configure the existing PyPI `pymab` project as a trusted publisher for
   repository `danielaLopes/pymab`, workflow `release.yml`, environment `pypi`.
4. The first crates.io publication must allocate the crate name. Create a scoped
   crates.io token that can publish only `pymab`, store it as the environment
   secret `CRATES_IO_TOKEN`, and leave the repository variable
   `CRATES_IO_TRUSTED_PUBLISHING` unset.
5. After that first crate exists, add a crates.io trusted-publishing rule for
   repository `danielaLopes/pymab`, workflow `release.yml`, environment
   `crates-io`. Set `CRATES_IO_TRUSTED_PUBLISHING=true`, remove the GitHub secret,
   and revoke the bootstrap token on crates.io. Later runs obtain a short-lived
   token through `rust-lang/crates-io-auth-action`.
6. Protect release tags matching `v*` and allow the Release Please App to create
   them.

## Prepare a release

Use Conventional Commit titles. `fix:` requests a patch, `feat:` a minor, and a
breaking-change marker a major release. Release Please opens the version-change
pull request; review the version it proposes rather than editing manifests by
hand.

Before merging that pull request, require all CI checks. They validate Rust 1.83,
all Python versions, the complete native policy registry, package versions,
native wheels, strict documentation, security audits, crates.io dry-run
publication, and benchmark compilation. The scheduled performance workflow
retains raw same-machine timing and RSS evidence.

## Publication transaction

Merging the release pull request creates the tag and GitHub Release. The release
workflow then:

1. Builds one `.crate`, one Python sdist, and CPython 3.11--3.14 wheels for Linux
   x86-64/aarch64, macOS x86-64/arm64, and Windows x86-64.
2. Tests every wheel on a runner with the matching operating system and CPU,
   including a forced Rust-backend experiment.
3. Inspects every archive, checks versions and matrix completeness, builds the
   packaged crate, and validates Python distribution metadata.
4. Queries both registries. It skips a publication only when that exact version
   already exists, allowing safe retry after a partial release.
5. Publishes crates.io first and waits until the version is visible. PyPI cannot
   start before that job succeeds.
6. Attaches verified artifacts and the benchmark evidence to the GitHub Release.

All artifact-verification jobs complete before either registry receives a
request. Approval of both protected environments is still required.

## Recovery and rollback

Registry artifacts are immutable and must never be overwritten. If only
crates.io succeeded, rerun the same workflow: the exact crate version is skipped
and the verified Python artifacts proceed. The inverse ordering is prevented by
the workflow dependency graph.

For a defective release, yank the affected crate version with `cargo yank` and
yank the PyPI release through its project administration page. Publish a new
version containing the fix; do not delete files, reuse a tag, or attempt to
replace an existing version.
