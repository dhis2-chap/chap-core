.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/flake8: ## check style with flake8
	flake8 climate_health tests

lint: lint/flake8 ## check style

test: ## run tests quickly with the default Python
	pytest

test-all: ## run pytest, doctests, examples
	mkdir -p logs/
	touch logs/rest_api.log
	touch logs/worker.log
	#./tests/test_docker_compose_flow.sh   # this runs pytests inside a docker container, can be skipped
	./tests/test_docker_compose_integration_flow.sh
	uv run pytest --durations=0 --cov=climate_health --cov-report html --run-slow
	uv run pytest --durations=0 --cov=climate_health --cov-report html --cov-append scripts/*_example.py
	#pytest --cov-report html --cov=chap_core --cov-append --doctest-modules chap_core/
	#cd docs_source && make doctest

coverage: ## check code coverage quickly with the default Python
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs_source/climate_health.rst
	rm -f docs_source/modules.rst
	sphinx-apidoc -o docs_source/ climate_health
	$(MAKE) -C docs_source clean
	$(MAKE) -C docs_source html
	$(BROWSER) docs_source/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs_source html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install -e .
