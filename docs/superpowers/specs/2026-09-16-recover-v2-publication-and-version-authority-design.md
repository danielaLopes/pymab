# Recover v2.0.0 Publication and Enforce One Version Authority

## Summary

PyMAB v2.0.0 already has a Git tag and a published GitHub release, but the release workflow stopped during artifact verification. Neither the Rust crate nor the Python distribution was published. This change will repair the release workflow, recover publication from the existing `v2.0.0` release, and remove a browser test that hardcodes the current release number.

The repository will keep one human-edited version authority: `[workspace.package].version` in `Cargo.toml`. Release Please will update that value. Package metadata, generated web assets, runtime assertions, and lock files will either derive from it or be checked against it.

## Goals

- Preserve the existing GitHub release and publish version 2.0.0 to crates.io and PyPI.
- Keep Rust 1.83 as the minimum supported Rust version.
- Support safe, explicit recovery of a failed publication for an existing GitHub release.
- Make browser tests verify the version that the application actually loaded instead of embedding a release number.
- Document and enforce `Cargo.toml` as the version source of truth.
- Keep registry publication idempotent so a retry skips a version that already exists.

## Non-goals

- Do not create another `v2.0.0` tag or GitHub release.
- Do not change version 2.0.0 to another release number.
- Do not raise the minimum supported Rust version to hide the dependency-resolution failure.
- Do not weaken the browser assertion to accept any semantic version.
- Do not merge the repair pull request automatically or merge it on the user's behalf.
- Do not redesign the Release Please versioning policy in this change.

## Current Failure

The `v2.0.0` release workflow built the Rust crate, Python source distribution, and native wheels. The `Verify complete release set` job then extracted the crate and ran `cargo test` directly against the unpacked library package.

The packaged library does not include a lock file. Running `cargo test` causes Cargo 1.83 to resolve development dependencies again. That fresh resolution selected `getrandom` 0.4.3, which requires Edition 2024 support from a newer Cargo. The failure is in the package's development-only dependency graph, not the published library dependency graph.

The repository's full test suite already tests the locked workspace. The unpacked-package check has a different purpose: prove that a downstream Rust 1.83 consumer can compile the public library and its normal dependencies. `cargo check --lib` models that purpose without resolving test and benchmark dependencies.

A separate Release Please pull request failed its browser job because `web/tests/runtime-smoke.spec.ts` expects the literal text `2.0.0`. The application correctly loaded and displayed the proposed 2.0.1 package version. The test bypassed the existing generated version flow and became stale as soon as Release Please proposed the next version.

## Version Authority

### Human-edited source

`Cargo.toml` owns the release version:

```toml
[workspace.package]
version = "2.0.0" # x-release-please-version
```

This is the only version value contributors should edit when preparing a release manually. In the normal flow, Release Please edits it from conventional commits.

### Derived and synchronized consumers

- Rust workspace crates use `version.workspace = true`.
- Maturin reads the Rust package version and provides dynamic Python package metadata.
- The web build packages the Python wheel and writes its version to `runtime-manifest.json`.
- The application reads that runtime manifest and displays the loaded PyMAB version.
- `Cargo.lock`, `uv.lock`, and `.release-please-manifest.json` are machine-maintained synchronization records.
- `scripts/check_versions.py` verifies that the workspace version, lock files, installed Python metadata, Release Please manifest, and optional native extension agree.

`.release-please-manifest.json` is required state for Release Please, but it is not a second human-maintained source of truth. CI must fail when it drifts from the workspace version.

### Browser assertion

The browser smoke test will request the generated `runtime-manifest.json`, read `pymabVersion`, open the developer panel, and require the displayed value to match the manifest exactly.

This is stronger than accepting any semantic version. It verifies the complete path from the packaged wheel to the generated manifest and rendered interface while remaining valid after any release bump.

## Release Workflow Design

### Supported triggers

The publication workflow will support both:

1. `release.published` for normal releases.
2. `workflow_dispatch` with a required `tag` input for recovery of an existing published GitHub release.

The metadata job will normalize both event types into one `target_tag` output. Every checkout and downstream job will use that output rather than reading `github.event.release.tag_name` directly.

### Manual recovery validation

Before building or accessing a publication environment, the metadata job will:

- Require the manual input to be a strict version tag such as `v2.0.0`.
- Confirm that the tag exists in the repository.
- Confirm that a published GitHub release exists for that exact tag.
- Check out the tag rather than the workflow branch.
- Read `[workspace.package].version` from the tagged `Cargo.toml`.
- Require the tag without its `v` prefix to equal the workspace version.

This prevents a manual run from publishing arbitrary branch contents or assigning artifacts to an unrelated release.

### Concurrency

The concurrency group will use the normalized target tag. A normal release and a manual recovery for the same tag therefore cannot publish concurrently.

### Artifact verification

The release continues to:

- Build exactly one Rust crate archive.
- Build one Python source distribution.
- Build and test the supported wheel matrix.
- Verify archive names, metadata, versions, and matrix coverage.
- Run strict `twine check` validation.
- Compare the crate bytes rebuilt in the publication job with the verified crate artifact.

The unpacked crate step changes from `cargo test` to:

```bash
cargo check --lib --manifest-path "target/package/extracted/pymab-${VERSION}/Cargo.toml"
```

Rust 1.83 remains active for this command. Full locked tests remain part of regular CI and package construction.

### Idempotent publication

The existing registry checks remain in place:

- Query crates.io for the exact package and version.
- Query PyPI for the exact package and version.
- Skip publication when the exact version already exists.
- Publish the Rust crate before the Python artifacts.
- Wait for crates.io visibility before proceeding to PyPI.

This supports a safe retry if one registry accepts a release and a later job fails.

### GitHub release assets

The verified crate, source distribution, and wheels will continue to be attached to the existing GitHub release. Recovery must target the release identified by `target_tag` and use overwrite-safe upload behavior so a repeated run does not create a second release.

## Environment Protection

The `crates-io` environment currently accepts `v*` tag references and requires review. The `pypi` environment also accepts `v*` tag references. A manually dispatched workflow is associated with its dispatch branch even though it checks out the validated tag, so the environments must also permit the default branch for this recovery path.

The repository settings will therefore allow `main` to deploy to both publication environments. This does not let arbitrary `main` contents publish because the workflow validates and checks out an existing published release tag before reaching either environment. The `crates-io` approval gate remains enabled, and no automatic merge or automatic approval is introduced.

The one-time `CRATES_IO_TOKEN` remains scoped to the `crates-io` environment. It will be used only if the trusted-publishing variable is not enabled. After the first successful crates.io publication, trusted publishing can be configured separately and the bootstrap token removed.

## Validation and CI Guards

The repair pull request will be checked with the repository's normal validation plus focused release checks:

- YAML and GitHub Actions syntax validation.
- `scripts/check_versions.py` and its tests.
- Workflow checks proving both triggers resolve the expected tag.
- A negative metadata test for a missing or mismatched tag.
- Rust workspace formatting, linting, locked tests, and package dry runs.
- Python formatting, linting, typing, tests, and packaging checks.
- Web formatting, linting, unit tests, production build, and Chromium end-to-end tests.
- A local unpacked-crate `cargo check --lib` under Rust 1.83.
- A browser assertion that compares the developer panel with `runtime-manifest.json`.

`scripts/check_versions.py` will continue to enforce the required release workflow structure. Its required fragments will be updated to cover manual recovery and the normalized target tag so later workflow edits cannot silently remove the recovery safety checks.

## Recovery Sequence

1. Implement and validate the repair on branch `recover-v2-publication`.
2. Push the branch and open a focused pull request.
3. Leave merging to the user.
4. After the pull request is merged, permit `main` in the `crates-io` and `pypi` environment deployment policies while retaining existing protections.
5. Dispatch the publication workflow from `main` with tag `v2.0.0`.
6. Wait for artifact construction and verification to finish.
7. Pause at the `crates-io` environment approval gate for user approval.
8. Publish and confirm `pymab` 2.0.0 on crates.io.
9. Publish and confirm `pymab` 2.0.0 on PyPI.
10. Confirm the artifacts attached to the existing GitHub release and report the registry links.

If the workflow fails after either registry accepts the release, rerun it with the same tag. Exact-version checks will skip completed registry publication and continue the remaining work.

## Expected Result

Version 2.0.0 is available from both package registries and remains tied to the existing `v2.0.0` source tag and GitHub release. Future release pull requests can change the workspace version without editing browser tests. Contributors have one clear version authority, and CI continues to detect drift in every machine-maintained copy.
