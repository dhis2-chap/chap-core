"""The CI path filter must only treat files as inert that no test or build step reads.

The `changes` job in the PR workflows skips the test jobs when a PR touches only the
paths negated in its `code` filter. If a test later starts reading one of those files,
a PR changing it would land untested, so the list is checked against tests/ and the
Makefile here. The block is also duplicated across workflows and must not drift.
"""

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

GATED_WORKFLOWS = [
    ".github/workflows/ci-test-external-models.yml",
    ".github/workflows/ci-test-python-install.yml",
]

# Negated patterns that are gated by their own workflow or tests rather than being inert.
COVERED_ELSEWHERE = {"docs/**", "mkdocs.yml", "charts/**"}


def _changes_job(workflow):
    content = yaml.safe_load((REPO_ROOT / workflow).read_text())
    return content["jobs"]["changes"]


def _filters(workflow):
    steps = _changes_job(workflow)["steps"]
    (filter_step,) = [step for step in steps if step.get("id") == "filter"]
    return yaml.safe_load(filter_step["with"]["filters"])


def _inert_paths(workflow):
    negated = [pattern[1:] for pattern in _filters(workflow)["code"] if pattern.startswith("!")]
    return [pattern for pattern in negated if pattern not in COVERED_ELSEWHERE]


def _readers():
    """Tracked test files plus the Makefile; local virtualenvs under fixtures are hidden dirs."""
    files = [
        path
        for path in (REPO_ROOT / "tests").rglob("*.py")
        if not any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts)
    ]
    files.append(REPO_ROOT / "Makefile")
    return [path for path in files if path != pathlib.Path(__file__).resolve()]


@pytest.mark.parametrize("workflow", GATED_WORKFLOWS)
def test_inert_paths_are_not_read_by_tests_or_makefile(workflow):
    for inert in _inert_paths(workflow):
        needle = inert.split("*")[0]
        assert needle, f"pattern {inert!r} is too broad to check by name"
        readers = [str(path.relative_to(REPO_ROOT)) for path in _readers() if needle in path.read_text()]
        assert not readers, (
            f"{inert!r} is skipped by the CI path filter in {workflow} but is referenced by "
            f"{readers}; either remove it from the filter or drop the reference"
        )


def test_changes_job_is_identical_across_workflows():
    first, *rest = [_changes_job(workflow) for workflow in GATED_WORKFLOWS]
    for other in rest:
        assert other == first
