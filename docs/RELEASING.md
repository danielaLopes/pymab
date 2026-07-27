# Releasing PyMAB

PyMAB releases are built and published by
[`release.yml`](../.github/workflows/release.yml). The workflow uses PyPI
Trusted Publishing, so it does not need a long-lived PyPI API token.

## One-time repository setup

1. In the GitHub repository settings, create an environment named `pypi`.
   Configure a required reviewer so every production publication needs explicit
   approval.
2. In the PyPI `pymab` project, add a GitHub Trusted Publisher with:
   - Owner: `danielaLopes`
   - Repository: `pymab`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Protect tags matching `v*` so only maintainers can create release tags.
4. After Trusted Publishing succeeds, remove the repository's
   `PYPI_API_TOKEN` secret and revoke the old token on PyPI.

## Publish a version

1. On a branch, update the version in `pyproject.toml`,
   `pymab/__init__.py`, and `docs/source/conf.py`.
2. Update `CHANGELOG.md`.
3. Open a pull request and wait for every required CI check to pass.
4. Merge the pull request.
5. Create a GitHub Release from `main` with a tag matching the package version,
   conventionally `vX.Y.Z` (for example, `v1.1.0`).
6. Publish the GitHub Release and approve the `pypi` environment deployment.

The release workflow rejects a tag whose version does not match
`pyproject.toml`. It builds the wheel and source distribution once, validates
their metadata, installs and imports the wheel on every supported Python
version, and publishes those same artifacts to PyPI.

Published PyPI files are immutable. If a release fails after any file reaches
PyPI, increment the version and publish a new release rather than reusing the
same tag or version.
