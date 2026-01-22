# Writing and building documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

The documentation is written in [Markdown format](https://www.markdownguide.org/basic-syntax/) which is simple to learn and easy to read.


## How to edit the documentation

All documentation is in the `docs` folder. The navigation structure is defined in `mkdocs.yml` at the project root.

Edit or add files in this directory to edit the documentation. When adding new files, remember to add them to the `nav` section in `mkdocs.yml`.


## How to build the documentation locally

From the project root, run:

```bash
make docs
```

Or directly with MkDocs:

```bash
uv run mkdocs build
```

The built documentation will be in the `site` directory. Open `site/index.html` to view it.


## Live preview

For a live preview that auto-reloads when you make changes:

```bash
uv run mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.


## API Documentation

API documentation is auto-generated from Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/). The API reference is in `docs/api/index.md`.


## Testing Code Examples (Doctests)

Code examples in documentation are automatically tested using [mktestdocs](https://github.com/koaning/mktestdocs) to ensure they remain correct and up-to-date.

### How it works

- **Python blocks** (` ```python `) are tested with `memory=True`, meaning code blocks within a file share state. Imports and variables defined in earlier blocks are available in later blocks.
- **Bash blocks** (` ```bash `) are tested as shell commands.
- **Console blocks** (` ```console `) are NOT tested and should be used for non-testable examples.

### Running doctests

```console
# Run fast documentation tests
make test-docs

# Run slow documentation tests (require models, data downloads)
make test-docs-slow

# Run all documentation tests
make test-docs-all
```

### Writing testable examples

**Aim to test as much as possible.** Prefer testable `python` and `bash` blocks over `console` blocks.

1. **Structure code blocks with shared state**: Put imports in earlier blocks, then usage in subsequent blocks that reference those imports.

2. **Use real module paths and class names**: Examples should use actual classes and functions from the codebase.

3. **Use `console` only when necessary**: When a code block cannot be tested (e.g., requires external services, would modify system state, or shows incomplete pseudocode), use `console` instead of `python`/`bash`. **Always add a comment explaining why** the block cannot be tested:

    ````markdown
    <!-- Cannot test: requires running Docker daemon -->
    ```console
    chap evaluate --model docker://my-model ...
    ```
    ````

4. **Avoid inline test data**: Use existing fixtures from `conftest.py` files when possible rather than creating new test data inline.

### Skipping files from testing

Some documentation files cannot be tested (e.g., they require Docker, external services, or would run destructive commands). To skip a file, add it to `SKIP_FILES` in `tests/test_documentation.py` with a comment explaining why:

```python
SKIP_FILES = [
    # These files have examples requiring external models, docker, or network access
    "docs/external_models/running_models_in_chap.md",
    # Files with commands that can't be safely tested
    "docs/contributor/testing.md",  # Contains pytest, make test-all commands
]
```

### Key files

- `tests/test_documentation.py` - Fast documentation tests
- `tests/test_documentation_slow.py` - Slow documentation tests (marked with `@pytest.mark.slow`)
