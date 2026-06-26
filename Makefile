.PHONY: clean coverage dist docs help install lint lint/flake8 check regen-plot-help force-restart restart chap-version architecture architecture-validate architecture-export
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@uv run python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ## remove all build, test, coverage and Python artifacts
	@echo ">>> Cleaning up"
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage coverage.xml htmlcov/
	@rm -rf dist/
	@rm -rf target/
	@rm -rf site/
	@rm -rf .cache

lint: ## check and fix code style with ruff, run type checking
	@echo "Linting code..."
	uv run ruff check --fix
	@echo "Formatting code..."
	uv run ruff format
	@echo "Type checking (mypy)..."
	uv run mypy
	@echo "Type checking (pyright)..."
	uv run pyright

check: check-alembic-heads ## non-mutating lint + type checks (used in CI)
	@echo "Ruff check..."
	uv run ruff check
	@echo "Ruff format check..."
	uv run ruff format --check
	@echo "Type checking (mypy)..."
	uv run mypy
	@echo "Type checking (pyright)..."
	uv run pyright

check-alembic-heads: ## fail if the alembic migration chain has more than one head
	@echo "Alembic head count..."
	@uv run python -c "from alembic.config import Config; from alembic.script import ScriptDirectory; heads = ScriptDirectory.from_config(Config('alembic.ini')).get_heads(); assert len(heads) == 1, f'Expected 1 alembic head, found {len(heads)}: {heads}'; print(f'OK: single head {heads[0]}')"

test: ## run tests quickly with minimal output (sequential; snappy startup)
	uv run pytest -q

test-parallel: ## run tests across all cores via pytest-xdist (faster wall time, slow startup)
	uv run pytest -q -n auto --dist loadfile

test-docs: ## run fast documentation code block tests
	uv run pytest tests/test_documentation.py -v

test-docs-slow: ## run slow documentation tests (requires --run-slow)
	uv run pytest tests/test_documentation_slow.py -v --run-slow

test-docs-all: ## run all documentation tests (fast + slow)
	uv run pytest tests/test_documentation.py tests/test_documentation_slow.py -v --run-slow

test-verbose: ## run tests with INFO level logging
	uv run pytest --log-cli-level=INFO -o log_cli=true -v

test-debug: ## run tests with DEBUG logging and SQL echo
	CHAP_DEBUG=true uv run pytest --log-cli-level=DEBUG -o log_cli=true -v -s -x

test-timed: ## run tests showing timing for 20 slowest tests
	uv run pytest -q --durations=20

test-all: ## run comprehensive test suite with examples and coverage
	mkdir -p target runs
	./tests/test_docker_compose_integration_flow.sh

	#./tests/test_docker_compose_flow.sh   # this runs pytests inside a docker container, can be skipped
	CHAP_DEBUG=true uv run pytest --log-cli-level=INFO -o log_cli=true -v --durations=0 --cov=climate_health --cov-report html --run-slow
	CHAP_DEBUG=true uv run pytest --log-cli-level=INFO -o log_cli=true -v --durations=0 --cov=climate_health --cov-report html --cov-append scripts/*_example.py

coverage: ## run tests with coverage reporting
	@echo ">>> Running tests with coverage"
	@uv run coverage run -m pytest -q
	@uv run coverage report
	@uv run coverage html
	@uv run coverage xml
	@echo "Coverage report: htmlcov/index.html"

docs: ## generate MkDocs HTML documentation (strict: warnings fail the build)
	NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict
	@echo "Docs: site/index.html"

dist: clean ## build source and wheel package
	uv build
	ls -l dist

install: clean ## sync dependencies and install package in development mode
	uv sync

regen-plot-help: ## regenerate chap_core/cli_endpoints/generated_plot_ids.py from @backtest_plot decorators
	@uv run python scripts/regenerate_plot_help.py

force-restart: ## tear down, rebuild, and start docker compose from scratch (WIPES VOLUMES including chap-db)
	docker compose -f compose.yml -f compose.chapkit.yml down -v && docker compose -f compose.yml -f compose.chapkit.yml build --no-cache && docker compose -f compose.yml -f compose.chapkit.yml up --remove-orphans

restart: ## soft restart docker compose (preserves volumes; rebuilds only on source changes)
	docker compose -f compose.yml -f compose.chapkit.yml down && docker compose -f compose.yml -f compose.chapkit.yml up -d --build --remove-orphans && $(MAKE) chap-version

chap-version: ## print the chap_core version running inside the chap container
	@docker compose -f compose.yml -f compose.chapkit.yml exec -T chap python -c 'import chap_core; print(f"chap_core running in container: {chap_core.__version__}")' 2>/dev/null || echo "chap container not running"

architecture: ## serve the interactive C4 architecture model (Structurizr) at http://localhost:8080
	docker run -it --rm -p 8080:8080 -v "$(CURDIR)/architecture:/usr/local/structurizr" structurizr/structurizr local

architecture-validate: ## validate the architecture model DSL (architecture/workspace.dsl)
	docker run --rm -v "$(CURDIR)/architecture:/work" -w /work structurizr/structurizr validate -workspace workspace.dsl

architecture-export: ## export all architecture diagrams to architecture/diagrams as PNG (needs port 8080 free; also pre-warms viewer thumbnails)
	@set -e; \
	docker rm -f chap-structurizr-export >/dev/null 2>&1 || true; \
	docker run -d --name chap-structurizr-export -p 8080:8080 -v "$(CURDIR)/architecture:/usr/local/structurizr" structurizr/structurizr local >/dev/null; \
	trap 'docker rm -f chap-structurizr-export >/dev/null 2>&1 || true' EXIT; \
	echo "Waiting for Structurizr to start..."; \
	for i in $$(seq 1 30); do curl -fsS -o /dev/null http://localhost:8080/ 2>/dev/null && break; sleep 2; done; \
	docker run --rm --network container:chap-structurizr-export \
		-e STRUCTURIZR_URL=http://localhost:8080 -e PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
		-v "$(CURDIR)/architecture:/work" -w /work mcr.microsoft.com/playwright:v1.55.0-noble \
		sh -c 'npm i playwright@1.55.0 --no-save --no-fund --no-audit --silent 2>/dev/null && node export-diagrams.js'; \
	echo "Diagrams exported to architecture/diagrams/"
