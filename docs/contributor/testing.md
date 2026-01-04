
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
source .venv/bin/activate
```

The quick test can be run simply by running `pytest` in the root folder of the project:

```bash
pytest
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
make test-all
```

To see what is actually being run, you can see what is specified under `test-all` in the Makefile.


## Some more details about integration tests

- The file `docker_db_flow.py` is important: This runs a lot of the db integration tests and tests for endpoints that are using the database and is run through a docker image when `make test-all` is run. Similarily the docker_flow.py runs some of the old endpoints (the db_flow.py is outdated and don't need to be run in future).
- Everything that the frontend uses should currently be added as tests in the docker_db_flow.py file
- In the future, we should ideally use the pytest framework for this integration test


## Documentation testing

We automatically test code blocks in our documentation to ensure examples stay up-to-date and work correctly. This uses [mktestdocs](https://github.com/koaning/mktestdocs) to extract and execute code blocks from markdown files.

### Two-tier testing system

Documentation tests are split into two tiers:

| Tier | Command | When to run | Duration |
|------|---------|-------------|----------|
| Fast | `make test-docs` | Every PR, pre-commit | ~10 seconds |
| Slow | `make test-docs-slow` | Weekly CI, manual | ~20 seconds |

**Fast tests** validate Python and bash code blocks in markdown files. They run automatically as part of the regular test suite.

**Slow tests** validate:
- JSON/YAML data format examples (schema validation)
- CLI help command output
- Python import statements from documentation
- Example dataset existence

### Running documentation tests

```console
# Run fast documentation tests
make test-docs

# Run slow documentation tests (requires --run-slow flag)
make test-docs-slow

# Run all documentation tests
make test-docs-all
```

### Writing testable code blocks

When adding code examples to documentation, use the appropriate language tag:

| Tag | Tested | Use for |
|-----|--------|---------|
| `python` | Yes | Python code that can be executed |
| `bash` | Yes | Shell commands that can be executed safely |
| `console` | No | Commands requiring external resources (docker, network, etc.) |
| `text` | No | Plain text output examples |
| `json` | No | JSON data examples (validated in slow tests via fixtures) |
| `yaml` | No | YAML configuration examples |

**Example - Testable bash command:**

````markdown
```bash
chap --help
```
````

**Example - Non-testable command (uses `console`):**

````markdown
```console
docker run -p 8000:8000 ghcr.io/dhis2-chap/chtorch:latest
```
````

### Skipping files

Some documentation files contain examples that cannot be safely tested (require external models, docker, network access, etc.). These are listed in `tests/test_documentation.py` in the `SKIP_FILES` list.

If you add a new documentation file that cannot be tested, add it to `SKIP_FILES` with a comment explaining why.

### Adding slow test fixtures

If you add new JSON/YAML examples to documentation that should be validated, add corresponding fixtures in `tests/fixtures/doc_test_data.py` and tests in `tests/test_documentation_slow.py`.

For example, to validate a new JSON schema:

```python
# In tests/fixtures/doc_test_data.py
MY_NEW_DATA_FORMAT = {
    "required_fields": ["field1", "field2"],
    "example": {"field1": "value1", "field2": "value2"}
}

# In tests/test_documentation_slow.py
@pytest.mark.slow
def test_my_new_data_format():
    assert "field1" in MY_NEW_DATA_FORMAT["example"]
```
