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
    chap eval --model-name docker://my-model --dataset-csv data.csv --output-file eval.nc
    ```
    ````

4. **Avoid inline test data**: Use existing fixtures from `conftest.py` files when possible rather than creating new test data inline.

5. **Render code output with markdown-exec**: To show code output in the built docs, add `exec="on" session="<name>" source="above"` to a Python code block. Blocks sharing the same `session` share state (imports, variables), similar to mktestdocs `memory=True`. Use `result="text"` for plain-text output (wraps in a code block), or omit it when the block prints markdown (e.g. `to_markdown()` tables) so it renders natively. Note: mktestdocs skips `exec="on"` blocks since the language tag is no longer plain `python`.

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


## Embedding CLI-generated plots

Some documentation pages include bash blocks that generate plot files (e.g., `chap plot-backtest`). These plots can be embedded inline in the rendered docs so readers see the actual output of the shown commands.

### How it works

1. Bash blocks in documentation run via mktestdocs during slow tests, generating output files (e.g., `.html` plots) in the project root.
2. `make generate-doc-assets` runs these tests and copies the generated plot files to `docs/generated/`.
3. An `<iframe>` tag in the markdown references the copied file, and `mkdocs build` includes it in the site.

The result is that the plot shown in the docs is the exact output of the CLI command displayed above it.

### Adding a new embedded plot

1. Add your `chap plot-backtest` bash block to the documentation as usual. The bash block will be tested by mktestdocs.

2. After the bash block, add an iframe pointing to `docs/generated/`:

    ```html
    <iframe src="../generated/my_plot.html" width="100%" height="500px" frameborder="0"></iframe>
    ```

    Adjust the `src` path relative to the markdown file's location in `docs/`.

3. Update the `generate-doc-assets` target in the `Makefile` to copy the new file:

    ```makefile
    @cp -f my_plot.html docs/generated/ 2>/dev/null || true
    ```

4. Add the filename to the cleanup line in both the `generate-doc-assets` and `clean` targets.

### Building docs with plots

```console
# Build docs without plots (fast)
make docs

# Build docs with embedded plots (slow, runs model evaluation)
make docs-with-plots
```
