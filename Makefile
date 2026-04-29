.PHONY: clean coverage dist docs help install lint lint/flake8 regen-plot-help test-chapkit-compose force-restart restart chap-version
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

test: ## run tests quickly with minimal output
	uv run pytest -q

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
	uv run mkdocs build --strict
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
