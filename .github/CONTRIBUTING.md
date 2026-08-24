# Contributing to PyMAB

PyMAB accepts bug reports, design discussions, documentation improvements, and
code contributions through GitHub. For a substantial or user-visible change,
open an issue before implementation so its statistical and API implications can
be agreed on first.

## Development setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended:

```console
git clone https://github.com/danielaLopes/pymab.git
cd pymab
make sync
```

`make sync` installs every development tool and all optional runtime features.
Its pip fallback requires a pip version that supports PEP 735 dependency groups.

## Quality checks

Run these commands from the repository root:

```console
make format             # verify formatting
make format-fix         # apply formatting
make lint               # Ruff and strict mypy
make test               # tests and branch coverage
make docs               # strict HTML, doctest, coverage, and snippets
make docs-linkcheck     # external links (network-dependent)
make security           # Bandit; opt into the network audit as shown below
PYMAB_RUN_NETWORK_AUDIT=1 make security
python -m build
twine check --strict dist/*
```

`make ci` runs the deterministic local CI subset. Add focused regression tests
for behavioral changes, document assumptions for statistical algorithms, and
update the migration guide for public API changes.

## Arcade development

The interactive website is a separate Node.js 24 project under ``web/``. It is
not included in the Python wheel and is deliberately absent from the Python
``sync`` and release targets.

```console
make web-sync        # npm ci
make web-format      # Prettier check
make web-lint        # ESLint and TypeScript
make web-test        # Vitest coverage
make web-build       # local wheel + self-hosted Pyodide + Vite
make web-e2e         # real browser/Pyodide tests
make web-ci          # deterministic non-browser web gates
```

``npm run prepare:python -- --clean`` rebuilds the ignored runtime directory
from scratch. If asset preparation fails, confirm that ``uv`` and Python 3 are
on ``PATH``, run ``npm ci`` to restore the pinned Pyodide package, and delete no
files outside ``web/.generated``. Hash mismatches are fatal by design; do not
bypass them or commit downloaded wheels.

## Pull requests

Create a focused branch from `main`, keep unrelated changes out of the commit,
and explain both the user-visible effect and the validation performed. The full
CI matrix checks Python 3.11 through 3.14, minimum dependencies, documentation,
examples, security, and the installed wheel.

## Releasing

Maintainers should follow the [release guide](../docs/RELEASING.md) to publish
new versions to PyPI.
