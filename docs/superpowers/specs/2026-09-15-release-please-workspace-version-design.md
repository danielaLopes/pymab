# Release Please workspace version design

## Problem

PyMAB stores its release version once in `[workspace.package]` and lets both
Rust crates inherit it with `version.workspace = true`. Release Please's Rust
strategy resolves those inherited values, then tries to write them back as
ordinary `[package].version` strings. Its TOML updater rejects the inherited
form and stops with `value at path package.version is not tagged`.

The earlier messages about unparseable commits are warnings caused by old
non-Conventional Commit subjects. They are not the failing condition and do
not justify rewriting repository history.

## Decision

Keep `[workspace.package].version` as the only authored package version. Use
Release Please's `simple` strategy for version selection, changelog updates,
the release pull request, the tag, and the GitHub release. Force its generic
updater for the root `Cargo.toml`, with `x-release-please-version` on the
workspace version line.

This preserves Cargo's supported workspace-inheritance model while avoiding
the incompatible Rust manifest updater. The existing targeted updates for the
two `Cargo.lock` package entries and the PyMAB entry in `uv.lock` remain part of
the release pull request.

## Configuration changes

1. Change the top-level release type in `release-please-config.json` from
   `rust` to `simple`.
2. Add `Cargo.toml` to `extra-files` with `type: generic`.
3. Annotate `[workspace.package].version` in `Cargo.toml` with
   `x-release-please-version`.
4. Keep the existing lockfile JSONPath updates unchanged.
5. Set `initial-version` to 2.0.0 while bootstrapping because the source and
   manifest reached 2.0.0 before a corresponding release tag existed.
6. Update the release documentation to explain why the generic updater is
   intentional and why member crates must continue inheriting the version.

## Release data flow

1. Conventional Commits determine the next semantic version.
   Until v2.0.0 is tagged, the configured initial version prevents Release
   Please from falling back to 1.0.0.
2. Release Please updates `.release-please-manifest.json` and `CHANGELOG.md`.
3. The generic updater replaces the annotated workspace version.
4. Targeted lockfile updaters replace the matching package versions.
5. CI validates the release pull request and confirms that every version
   source agrees.
6. The user may merge the release pull request. Automation must never merge a
   pull request into `main` or enable auto-merge.

## Failure handling

- If the annotation is removed, the configuration test must fail before a
  release run reaches GitHub.
- If a lockfile path or package name changes, version-consistency checks must
  reject the release pull request.
- Historical non-Conventional Commit messages remain warnings. New changes
  should use Conventional Commit subjects so they can be classified correctly.

## Validation

- Parse `release-please-config.json` and all workflow YAML files.
- Run `actionlint` against the workflows.
- Add a repository check that asserts the `simple` strategy, the forced generic
  `Cargo.toml` updater, the version annotation, and both inherited member
  versions remain configured together.
- Run the pinned Release Please CLI in dry-run mode against the pushed test
  branch and assert that it proposes the next version without touching either
  member crate's `version.workspace = true` declaration.
- Confirm the proposed changes include `Cargo.toml`, `Cargo.lock`, `uv.lock`,
  `.release-please-manifest.json`, and `CHANGELOG.md`.
- Run the repository's existing version-consistency, package, Rust, and Python
  CI checks.

## Non-goals

- Do not give the two Rust crates independent release versions.
- Do not duplicate literal versions in member manifests.
- Do not rewrite old commit messages.
- Do not merge or enable auto-merge for any pull request targeting `main`.
