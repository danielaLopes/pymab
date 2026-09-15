# Complete release publication

## Problem

The Release Please workflow now uses the CLI because the GitHub Action wrapper
drops `release-as` in manifest mode. The workflow invokes only `release-pr`, so
it can create and update release pull requests but cannot turn a merged release
pull request into a tag and GitHub release. Pull request 17 is merged at 2.0.0,
yet `v2.0.0` does not exist and the registry publication workflow has not run.

## Design

Use the pinned Release Please CLI for both phases in their required order:

1. Run `github-release` against `main`. This detects a merged, untagged release
   pull request and creates the matching tag and GitHub release.
2. Query GitHub for the workspace version's tag after `github-release` has run.
   Only retain the existing bootstrap override if the remote tag is still
   absent. Treat only an HTTP 404 as absence; fail closed on authentication,
   rate-limit, or network errors.
3. Run `release-pr` with that self-disabling version override. This maintains
   the next release proposal after any pending release is finalized without
   reopening a proposal for the release that was just tagged.

Both commands use the same short-lived GitHub App token and the same manifest
configuration. The GitHub release emits the existing `release: published`
event, which starts `.github/workflows/release.yml`.

## Rust and Python publication

The release workflow keeps one version across the public `pymab` Rust crate and
the Python distribution. It packages and verifies all artifacts before registry
writes, publishes the crate first, waits for crates.io visibility, and then
publishes Python artifacts to PyPI. The private `pymab-python` bindings crate
remains `publish = false`.

The first crates.io publication requires a narrowly scoped API token because a
trusted publisher cannot be attached before the crate exists. The repository
owner must create the `crates-io` GitHub environment and add the token as
`CRATES_IO_TOKEN`. After the first publish, the owner can configure trusted
publishing for `release.yml`, set `CRATES_IO_TRUSTED_PUBLISHING=true`, and revoke
the bootstrap token.

## Safety and failure behavior

- Keep Release Please pinned to 17.6.0 for both commands.
- Run `github-release` before `release-pr` and stop immediately if it fails.
- Keep registry publishing exclusively on the GitHub `release: published`
  event.
- Keep protected environments and artifact verification between tag creation
  and registry writes.
- Never publish the private PyO3 bindings crate.
- Do not create the tag or publish registry artifacts while validating this
  feature branch. Use dry-run and package checks only.

## Validation

1. Validate workflow YAML with `actionlint` and a YAML parser.
2. Extend the repository version gate to require both Release Please CLI phases
   in the correct order.
3. Run `github-release --dry-run` remotely and confirm it identifies 2.0.0.
4. Run `release-pr --dry-run` remotely and confirm normal proposal behavior.
5. Run `cargo publish --dry-run -p pymab --locked`.
6. Build and verify the Python distribution without uploading it.
