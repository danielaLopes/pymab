# Releasing PyMAB

Release Please maintains the version and changelog in a release pull request.
Merging that pull request creates a `vX.Y.Z` tag and a GitHub Release. The
[`release.yml`](../.github/workflows/release.yml) workflow then builds, tests,
and publishes that release with PyPI Trusted Publishing.

## One-time repository setup

1. Create a GitHub App installed only on this repository with read/write
   access to contents, issues, and pull requests.
2. Add the App ID as the Actions repository variable
   `RELEASE_PLEASE_APP_ID`.
3. Add the complete GitHub App private key as the Actions repository secret
   `RELEASE_PLEASE_PRIVATE_KEY`.
4. In the GitHub repository settings, create an environment named `pypi`.
   Configure a required reviewer so every production publication needs
   explicit approval.
5. In the PyPI `pymab` project, add a GitHub Trusted Publisher with:
   - Owner: `danielaLopes`
   - Repository: `pymab`
   - Workflow: `release.yml`
   - Environment: `pypi`
6. Protect tags matching `v*` while allowing the Release Please GitHub App to
   create release tags.
7. After Trusted Publishing succeeds, remove the repository's
   `PYPI_API_TOKEN` secret and revoke the old token on PyPI.

## Prepare changes for a release

Use a Conventional Commit title for each squash-merged pull request:

- `fix: ...` requests a patch release.
- `feat: ...` requests a minor release.
- `feat!: ...` or a `BREAKING CHANGE:` footer requests a major release.

After a releasable change reaches `main`, Release Please opens or updates one
release pull request. It updates `CHANGELOG.md`, `.release-please-manifest.json`,
`pyproject.toml` and `uv.lock`. The package and documentation read that metadata
at runtime, so no duplicate version declarations need updating.
Do not edit those versions manually.

## Publish a version

1. Review the open Release Please pull request.
2. Wait for every required CI check to pass.
   The documentation matrix performs clean HTML and doctest builds on the
   minimum and latest supported Python versions, enforces API documentation
   coverage, executes README snippets, and uploads the rendered site. External
   link checking runs separately as a non-blocking signal because remote sites
   can fail transiently.
3. Merge the release pull request when the accumulated changes are ready.
4. Release Please creates the matching tag and GitHub Release.
5. Open the `Publish to PyPI` workflow run and approve its `pypi` environment
   deployment.
6. Verify the new version on PyPI.

The release workflow rejects a tag whose version does not match
`pyproject.toml`. It builds the wheel and source distribution once, validates
their metadata, installs and imports the wheel on every supported Python
version, and publishes those same artifacts to PyPI.

Published PyPI files are immutable. If a release fails after any file reaches
PyPI, increment the version and publish a new release rather than reusing the
same tag or version.

## Version 2 release

The v2 migration is intentionally breaking. Review the migration guide and use
a breaking Conventional Commit so Release Please preserves major-version
semantics for subsequent releases.

For an exceptional manual override, run the `Release Please` workflow from the
Actions tab and provide an exact semantic version in the `release_as` input.
