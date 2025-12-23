"""Test code blocks in documentation files."""

import pathlib

import pytest

from mktestdocs import check_md_file

# Files to skip entirely (contain only non-testable examples)
SKIP_FILES = [
    # These files have examples requiring external models, docker, or network access
    "docs/chap-cli/evaluation-workflow.md",
    "docs/external_models/chapkit.md",
    "docs/external_models/chap_evaluate_examples.md",
    "docs/external_models/running_models_in_chap.md",
    "docs/modeling-app/installation.md",
    "docs/contributor/chap-contributor-setup.md",
    "docs/contributor/getting_started.md",
    "docs/chap-cli/chap-core-cli-setup.md",
    "docs/chap-cli/downloaded_json_data.md",
    "docs/chap-cli/using-pa-with-cli.md",
    "docs/chap-cli/earth-engine-auth.md",
    "docs/webapi/docker-compose-doc.md",
    "docs/contributor/windows_contributors.md",
    "docs/contributor/database_migrations.md",
    "docs/contributor/writing_building_documentation.md",
    # Design documents and planning docs
    "docs/contributor/evaluation_abstraction.md",
    "docs/contributor/preference_learning.md",
    # Files with bash prompts ($) or outdated Python imports
    "docs/contributor/testing.md",
    "docs/external_models/surplus_after_refactoring.md",
]


def get_doc_files():
    """Get all markdown files in docs/ that are not in SKIP_FILES."""
    all_files = list(pathlib.Path("docs").glob("**/*.md"))
    return [f for f in all_files if str(f) not in SKIP_FILES]


@pytest.mark.parametrize("fpath", get_doc_files(), ids=str)
def test_docs_python(fpath):
    """Test Python code blocks in documentation."""
    check_md_file(fpath=fpath, lang="python")


@pytest.mark.parametrize("fpath", get_doc_files(), ids=str)
def test_docs_bash(fpath):
    """Test bash code blocks in documentation."""
    check_md_file(fpath=fpath, lang="bash")
