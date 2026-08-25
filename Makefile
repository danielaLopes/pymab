.PHONY: audit benchmark benchmark-check benchmark-smoke ci docs docs-coverage docs-doctest docs-html docs-linkcheck \
	docs-snippets format format-fix lint llm-security native rust-format \
	rust-format-fix rust-lint rust-test security sync test test-ci

UV ?= uv
CARGO ?= cargo
PYTHON ?= python3.12
PYTHON_VERSION ?= 3.12
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR
RUN_TOOL = sh scripts/run_tool.sh
MATURIN = $(RUN_TOOL) maturin
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

native:
	$(MATURIN) develop --manifest-path crates/pymab-python/Cargo.toml

benchmark:
	$(RUN_TOOL) python -m benchmarks.run_backends --all --output benchmarks/results/local.json
	$(RUN_TOOL) python -m benchmarks.report benchmarks/results/local.json --check-thresholds

benchmark-check:
	$(RUN_TOOL) python -m benchmarks.report benchmarks/results/local.json --check-thresholds

benchmark-smoke:
	$(RUN_TOOL) python -m benchmarks.run_backends --all --horizon 12 --n-replicates 2 \
		--repetitions 1 --output benchmarks/results/smoke.json

rust-format:
	$(CARGO) fmt --all --check

rust-format-fix:
	$(CARGO) fmt --all

rust-lint:
	$(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings

rust-test:
	$(CARGO) test --workspace --all-features --locked

lint:
	$(RUN_TOOL) ruff check .
	$(RUN_TOOL) mypy src/pymab

test:
	$(RUN_TOOL) pytest --cov-fail-under=92

test-ci:
	$(RUN_TOOL) pytest --cov-fail-under=92

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
