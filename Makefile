.PHONY: clean coverage dist docs docs-with-plots generate-doc-assets help install lint lint/flake8 test-chapkit-compose
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
	@rm -f comparison_doctest.csv comparison_specific_doctest.csv eval_doctest.nc plot_doctest.html metrics_doctest.csv

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
	CHAP_DEBUG=true uv run chap evaluate --model-name https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --prediction-length 3
	CHAP_DEBUG=true uv run chap evaluate --model-name external_models/naive_python_model_with_mlproject_file_and_docker/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --model-configuration-yaml external_models/naive_python_model_with_mlproject_file_and_docker/example_model_configuration.yaml

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

generate-doc-assets: ## generate plots for documentation (slow, runs model evaluation)
	uv run pytest tests/test_documentation_slow.py::TestSlowDocumentationBash -v --run-slow
	@mkdir -p docs/generated
	@cp -f plot_doctest.html docs/generated/ 2>/dev/null || true
	@rm -f comparison_doctest.csv comparison_specific_doctest.csv eval_doctest.nc plot_doctest.html metrics_doctest.csv

docs: ## generate MkDocs HTML documentation
	uv run mkdocs build
	@echo "Docs: site/index.html"

docs-with-plots: generate-doc-assets docs ## generate documentation with embedded plots

dist: clean ## build source and wheel package
	uv build
	ls -l dist

install: clean ## sync dependencies and install package in development mode
	uv sync
