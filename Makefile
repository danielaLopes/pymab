.PHONY: audit ci docs docs-coverage docs-doctest docs-html docs-linkcheck \
	docs-snippets format format-fix lint llm-security security sync test test-ci demo-test \
	web-sync web-format web-lint web-test web-build web-e2e web-ci

UV ?= uv
PYTHON ?= python3.12
PYTHON_VERSION ?= 3.12
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR
RUN_TOOL = sh scripts/run_tool.sh
DOCS_SOURCE = docs/source
DOCS_BUILD = docs/build
SPHINX_STRICT_FLAGS = -W --keep-going -n -E -a
DOCS_COVERAGE_MIN ?= 100
UV_EXPORT_FLAGS ?= --frozen --all-extras --all-groups --no-emit-project --no-hashes
PIP_AUDIT_FLAGS ?= --strict
ifeq ($(CI),true)
PYMAB_PREFER_VENV ?= 0
else
PYMAB_PREFER_VENV ?= 1
endif
export PYMAB_PREFER_VENV

ci: format lint security test-ci

sync:
	@$(UV) sync --python $(PYTHON_VERSION) --dev --all-extras || { \
		echo "uv sync failed; falling back to venv + pip"; \
		$(PYTHON) -m venv .venv; \
		. .venv/bin/activate; \
		python -m pip install --upgrade pip; \
		python -m pip install --group dev -e ".[plot,docs,analysis,bayes]"; \
	}

format:
	$(RUN_TOOL) ruff format --check .

format-fix:
	$(RUN_TOOL) ruff format .

lint:
	$(RUN_TOOL) ruff check .
	$(RUN_TOOL) mypy src/pymab web/python

test:
	$(RUN_TOOL) pytest --cov-fail-under=92

test-ci:
	$(RUN_TOOL) pytest --cov-fail-under=92

demo-test:
	$(RUN_TOOL) pytest -o addopts= tests/demo --cov=web/python/pymab_demo \
		--cov-branch --cov-report=term-missing --cov-fail-under=95

security:
	$(RUN_TOOL) bandit -r src/pymab --severity-level low --confidence-level medium
	@if [ "$${PYMAB_RUN_NETWORK_AUDIT:-0}" = "1" ]; then \
		$(MAKE) audit; \
	else \
		echo "Skipping pip-audit dependency audit; set PYMAB_RUN_NETWORK_AUDIT=1 to enable networked auditing."; \
	fi

audit:
	@tmp_file=$$(mktemp); \
	trap 'rm -f "$$tmp_file"' EXIT; \
	$(UV) export --quiet $(UV_EXPORT_FLAGS) --format requirements.txt --output-file "$$tmp_file" >/dev/null; \
	$(RUN_TOOL) pip-audit --requirement "$$tmp_file" $(PIP_AUDIT_FLAGS)

llm-security:
	$(RUN_TOOL) python scripts/llm_security_review.py

docs: docs-html docs-doctest docs-coverage docs-snippets

docs-html:
	$(RUN_TOOL) sphinx-build $(SPHINX_STRICT_FLAGS) -b html \
		$(DOCS_SOURCE) $(DOCS_BUILD)/html

docs-doctest:
	$(RUN_TOOL) sphinx-build $(SPHINX_STRICT_FLAGS) -b doctest \
		$(DOCS_SOURCE) $(DOCS_BUILD)/doctest

docs-coverage:
	$(RUN_TOOL) sphinx-build $(SPHINX_STRICT_FLAGS) -b coverage \
		$(DOCS_SOURCE) $(DOCS_BUILD)/coverage
	$(RUN_TOOL) python scripts/check_docs_coverage.py \
		$(DOCS_BUILD)/coverage/python.txt --minimum $(DOCS_COVERAGE_MIN)

docs-snippets:
	$(RUN_TOOL) python scripts/check_readme_snippets.py README.md

docs-linkcheck:
	$(RUN_TOOL) sphinx-build $(SPHINX_STRICT_FLAGS) -b linkcheck \
		$(DOCS_SOURCE) $(DOCS_BUILD)/linkcheck

web-sync:
	cd web && npm ci

web-format:
	cd web && npm run format:check

web-lint:
	cd web && npm run lint && npm run typecheck

web-test:
	cd web && npm run test:coverage

web-build:
	cd web && npm run build

web-e2e:
	cd web && npm run e2e

web-ci: web-format web-lint web-test web-build
