.PHONY: audit ci docs format format-fix lint llm-security security sync test test-ci

UV ?= uv
PYTHON ?= python3.12
PYTHON_VERSION ?= 3.12
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR
RUN_TOOL = sh scripts/run_tool.sh
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
		python -m pip install -e ".[dev,plot,docs]"; \
	}

format:
	$(RUN_TOOL) ruff format --check .

format-fix:
	$(RUN_TOOL) ruff format .

lint:
	$(RUN_TOOL) ruff check .
	$(RUN_TOOL) mypy pymab

test:
	$(RUN_TOOL) pytest --cov-fail-under=75

test-ci:
	$(RUN_TOOL) pytest --cov-fail-under=75

security:
	$(RUN_TOOL) bandit -r pymab --severity-level low --confidence-level medium
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

docs:
	$(RUN_TOOL) sphinx-build docs/source docs/build
