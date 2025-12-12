# Plan: Documentation Code Block Testing

This document outlines a plan for implementing automated testing of code blocks in markdown documentation.

## Goal

Ensure code examples in documentation stay up-to-date and working as the codebase evolves.

## Recommended Tool: mktestdocs

[mktestdocs](https://github.com/koaning/mktestdocs) is designed for mkdocs and supports both Python and bash code blocks.

### Installation

```bash
pip install mktestdocs
```

Or add to `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "mktestdocs",
    # ... other doc dependencies
]
```

## Implementation

### 1. Create Test File

Create `tests/test_documentation.py`:

```python
import pathlib
import pytest
from mktestdocs import check_md_file

# Test all markdown files
@pytest.mark.parametrize(
    'fpath',
    pathlib.Path("docs").glob("**/*.md"),
    ids=str
)
def test_docs_python(fpath):
    """Test Python code blocks in documentation."""
    check_md_file(fpath=fpath, lang="python")

@pytest.mark.parametrize(
    'fpath',
    pathlib.Path("docs").glob("**/*.md"),
    ids=str
)
def test_docs_bash(fpath):
    """Test bash code blocks in documentation."""
    check_md_file(fpath=fpath, lang="bash")
```

### 2. Handle Non-Testable Code Blocks

Since mktestdocs doesn't have a built-in skip directive, use one of these approaches:

#### Option A: Use a different language tag for non-testable blocks

```markdown
<!-- Testable -->
```bash
chap --help
```

<!-- Not testable - use 'console' or 'shell' instead -->
```console
chap evaluate2 --model-name https://github.com/...
```
```

#### Option B: Create wrapper that filters files

```python
# tests/test_documentation.py
SKIP_FILES = [
    "docs/chap-cli/evaluation-workflow.md",  # Has external dependencies
]

@pytest.mark.parametrize(
    'fpath',
    [f for f in pathlib.Path("docs").glob("**/*.md") if str(f) not in SKIP_FILES],
    ids=str
)
def test_docs_bash(fpath):
    check_md_file(fpath=fpath, lang="bash")
```

#### Option C: Use pytest-phmdoctest for skip support

If skip directives are needed, use [pytest-phmdoctest](https://pypi.org/project/pytest-phmdoctest/) instead:

```bash
pip install pytest-phmdoctest
```

Add skip comments in markdown:

```markdown
<!--phmdoctest-skip-->
```bash
# This block won't be tested
chap evaluate2 --model-name https://github.com/...
```
```

### 3. Create Testable Examples

For CLI documentation, add simple testable commands alongside full examples:

```markdown
## Verify Installation

```bash
# Testable: verify CLI is installed
chap --help
chap evaluate2 --help
chap export-metrics --help
```

## Full Example

```console
# Full workflow (requires external models)
chap evaluate2 --model-name https://github.com/dhis2-chap/...
```
```

### 4. Integration Test with Example Data

Create a separate integration test that runs the full workflow:

```python
# tests/integration/test_cli_workflow.py
import subprocess
import tempfile
from pathlib import Path

import pytest

EXAMPLE_DATA = Path("example_data/laos_subset.csv")

@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_workflow():
    """Test the complete evaluation workflow from docs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_nc = Path(tmpdir) / "eval.nc"
        output_csv = Path(tmpdir) / "metrics.csv"

        # Run evaluate2 with a fast internal model
        result = subprocess.run([
            "chap", "evaluate2",
            "--model-name", "naive_model",  # Use internal fast model
            "--dataset-csv", str(EXAMPLE_DATA),
            "--output-file", str(output_nc),
            "--backtest-params.n-splits", "2",
        ], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
        assert output_nc.exists()

        # Run export-metrics
        result = subprocess.run([
            "chap", "export-metrics",
            "--input-files", str(output_nc),
            "--output-file", str(output_csv),
        ], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
        assert output_csv.exists()
```

## CI Configuration

Add to GitHub Actions workflow:

```yaml
- name: Test documentation
  run: |
    pip install mktestdocs
    pytest tests/test_documentation.py -v
```

## Checklist

- [ ] Install mktestdocs (or pytest-phmdoctest)
- [ ] Create `tests/test_documentation.py`
- [ ] Update docs to use `console` tag for non-testable examples
- [ ] Add simple testable `bash` examples (--help commands)
- [ ] Create integration test for full workflow
- [ ] Add to CI pipeline
- [ ] Document the testing approach in CONTRIBUTING.md

## References

- [mktestdocs](https://github.com/koaning/mktestdocs)
- [pytest-phmdoctest](https://pypi.org/project/pytest-phmdoctest/)
- [pytest-markdown-docs](https://github.com/modal-labs/pytest-markdown-docs)
