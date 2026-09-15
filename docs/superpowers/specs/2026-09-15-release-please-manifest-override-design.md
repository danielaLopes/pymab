# Release Please manifest override repair

## Problem

PyMAB uses Release Please in manifest mode. The repository source and manifest
are at 2.0.0, but the existing release pull request still proposes 1.0.0. The
workflow passes `release-as: 2.0.0` to Release Please Action v5, yet the action
does not forward that value when it loads a manifest configuration. The action
run succeeds while updating the release pull request with the wrong version.

The browser test that expects 2.0.0 is correct. It catches the generated release
branch packaging a 1.0.0 wheel and must remain unchanged.

## Design

Replace the Release Please Action step with the pinned Release Please CLI. Keep
the existing GitHub App authentication, manifest files, version discovery, and
workflow permissions. Build the CLI arguments in a shell array and add
`--release-as` only when the workflow produced a non-empty override.

The CLI reads `release-please-config.json` and `.release-please-manifest.json`
from the target branch. Unlike the action wrapper, it forwards `--release-as`
in manifest mode. On the next run against `main`, it will update the existing
release branch and pull request to 2.0.0. Once a `v2.0.0` tag exists, the current
version-discovery step stops adding the override, so normal Conventional Commit
versioning resumes.

## Safety and failure behavior

- Pin the CLI to 17.6.0, matching the version bundled by the current action.
- Keep the GitHub App token scoped to contents, issues, and pull requests.
- Do not close, merge, or delete the existing release pull request from this
  feature branch.
- Let shell strict mode and the CLI exit code fail the workflow if argument
  construction or release generation fails.
- Keep the browser version assertion unchanged so a future downgrade remains a
  CI failure.

## Validation

1. Run workflow syntax validation and `actionlint`.
2. Run the repository version gate.
3. Run Release Please 17.6.0 remotely with `--dry-run` and
   `--release-as=2.0.0` against this branch.
4. Confirm the proposal is 2.0.0 and includes the root Cargo manifest and both
   lockfiles.
5. Run the web production build and the Chromium runtime smoke test.

