
# Testing while developing

We rely on having most of the codebase well tested, so that we can be confident that new changes don't break stuff. Although there is some overhead writing tests,
having good tests makes developing and pushing new features much faster.

We use [pytest](https://docs.pytest.org/en/6.2.x/) as our testing framework. To run the tests.

The tests are split into quick tests that one typically runs often while developing and more comprehensive tests that are run less frequently.

We recomment the following:

- Run the quick tests frequently while developing. Ideally have a shortcut or easy way to run these through your IDE.
- Run the comprehensive tests before pushing new code. These are also run automatically on Github actions, but we want to try to avoid these failing there, so we try to discover
problems ideally before pushing new code.


## The quick tests

First make sure you have activated your local development environment:

```bash
$ source .venv/bin/activate
```

The quick test can be run simply by running `pytest` in the root folder of the project:

```bash
$ pytest
```

All tests should pass. If you write a new test and it is not passing for some reason (e.g. the functionalit you are testing is not implemented yet),
you can mark the test as `xfail` by adding the `@pytest.mark.xfail` decorator to the test function. This will make the test not fail the test suite.

```python
import pytest

@pytest.mark.xfail
def test_my_function():
    assert False
```

If you have slow tests that you don't want to be included every time you run pytest, you can mark them as slow.

```python
import pytest

@pytest.mark.slow
def test_my_slow_function():
    assert True
```

Such tests are not included when running pytest, but included when running `pytest --run-slow` (see below).

## The comprehensive tests

The comprehensive tests include the quick tests (see above) in addition to:

- slow tests (marked with `@pytest.mark.slow`). 
- Some tests for the integration with various docker containers 
- Pytest run on all files in the scripts directory that contains `_example` in the file name. The idea is that one can put code examples here that are then automatically tested.
- Docetests (all tests and code in the documentation)

The comprehensive tests are run by running this in the root folder of the project:

```bash
$ make test-all
```

To see what is actually being run, you can see what is specified under `test-all` in the Makefile.


## Some more details about integration tests

- The file `docker_db_flow.py` is important: This runs a lot of the db integration tests and tests for endpoints that are using the database and is run through a docker image when `make test-all` is run. Similarily the docker_flow.py runs some of the old endpoints (the db_flow.py is outdated and don't need to be run in future).
- Everything that the frontend uses should currently be added as tests in the docker_db_flow.py file
- In the future, we should ideally use the pytest framework for this integration test
