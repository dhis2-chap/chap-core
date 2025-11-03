.PHONY: clean coverage dist docs help install lint lint/flake8 test-chapkit-compose
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
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@rm -rf .coverage coverage.xml htmlcov/
	@rm -rf .tox/
	@rm -rf dist/ build/ *.egg-info .eggs/

lint:
	@echo "Linting code..."
	uv run ruff check --fix
	@echo "Formatting code..."
	uv run ruff format

test: ## run tests quickly with the default Python
	uv run pytest

	@rm model_config.yaml
	@rm example_data/debug_model/model_configuration_for_run.yaml

test-chapkit-compose: ## test docker compose with chapkit models
	./tests/test-chapkit-compose.sh

test-all: ## run pytest, doctests, examples
	./tests/test_docker_compose_integration_flow.sh
	uv run chap evaluate --model-name https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --prediction-length 3
	uv run chap evaluate --model-name external_models/naive_python_model_with_mlproject_file_and_docker/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --model-configuration-yaml external_models/naive_python_model_with_mlproject_file_and_docker/example_model_configuration.yaml

	#./tests/test_docker_compose_flow.sh   # this runs pytests inside a docker container, can be skipped
	uv run pytest --durations=0 --cov=climate_health --cov-report html --run-slow
	uv run pytest --durations=0 --cov=climate_health --cov-report html --cov-append scripts/*_example.py
	#pytest --cov-report html --cov=chap_core --cov-append --doctest-modules chap_core/
	#cd docs_source && make doctest
	@rm report.csv
	@rm predictions.csv
	@rm model_config.yaml
	@rm model.pkl
	@rm example_data/debug_model/model_configuration_for_run.yaml
	@rm evaluation_report.pdf

coverage: ## run tests with coverage reporting
	@echo ">>> Running tests with coverage"
	@uv run coverage run -m pytest -q
	@uv run coverage report
	@uv run coverage html
	@uv run coverage xml
	@rm test.csv
	@rm example_data/debug_model/model_configuration_for_run.yaml
	@echo "Coverage report: htmlcov/index.html"

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs_source/chap_core.rst
	rm -f docs_source/modules.rst
	uv run sphinx-apidoc -o docs_source/ chap_core
	$(MAKE) -C docs_source clean
	$(MAKE) -C docs_source html
	@echo "Docs: docs_source/_build/html/index.html"

dist: clean ## builds source and wheel package
	uv build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	uv sync
