"""Tests for the change-scope composite action that gates CI jobs.

The action decides whether a pull request needs the full test suite, only the
fast jobs, or no test jobs at all. It is plain bash embedded in a workflow
action, so these tests run that exact script with a stubbed ``gh`` binary.
"""

import os
import stat
import subprocess

import pytest
import yaml

# The action's script is bash and only ever runs on ubuntu runners, so the
# tests that execute it need a shell the Windows leg does not have.
needs_bash = pytest.mark.skipif(os.name == "nt", reason="The change-scope action is bash and runs on ubuntu only")


@pytest.fixture
def repo_root(tests_path):
    return tests_path.parent


@pytest.fixture
def classify_script(repo_root):
    """The bash script embedded in the change-scope action."""
    action = yaml.safe_load((repo_root / ".github" / "actions" / "change-scope" / "action.yml").read_text())
    (step,) = [s for s in action["runs"]["steps"] if s.get("id") == "classify"]
    return step["run"]


def run_classify(script, tmp_path, changed_files, event_name="pull_request", skip_label="false"):
    """Run the action script against a stubbed list of changed files."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh_stub = bin_dir / "gh"
    gh_stub.write_text('#!/usr/bin/env bash\nprintf "%s" "$CHANGED_FILES_STUB"\n')
    gh_stub.chmod(gh_stub.stat().st_mode | stat.S_IEXEC)

    output_file = tmp_path / "github_output"
    output_file.write_text("")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CHANGED_FILES_STUB": "".join(f"{f}\n" for f in changed_files),
        "GITHUB_OUTPUT": str(output_file),
        "RUNNER_TEMP": str(tmp_path),
        "GH_TOKEN": "stub",
        "EVENT_NAME": event_name,
        "PR_NUMBER": "1",
        "REPOSITORY": "dhis2-chap/chap-core",
        "SKIP_LABEL": skip_label,
    }
    subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True)

    outputs = dict(line.split("=", 1) for line in output_file.read_text().splitlines() if "=" in line)
    return outputs["scope"]


@needs_bash
@pytest.mark.parametrize(
    "changed_files, expected",
    [
        # Source changes need everything.
        (["chap_core/api.py"], "full"),
        (["tests/test_cli.py"], "full"),
        (["pyproject.toml"], "full"),
        (["charts/chap/values.yaml"], "full"),
        # Markdown under docs/ has code blocks that test_documentation.py runs,
        # so the fast test job must stay while the heavy suite can go.
        (["docs/chap-cli/minimalist-example.md"], "docs"),
        (["docs/_static/screenshot.png"], "docs"),
        # Nothing executes these, so no test job can be affected.
        (["README.md"], "inert"),
        (["charts/README.md"], "inert"),
        ([".env.example"], "inert"),
        (["LICENSE"], "inert"),
        ([".github/ISSUE_TEMPLATE/bug.yml"], "inert"),
        # A mixed pull request falls back to the widest scope it contains,
        # in either file order.
        (["README.md", "chap_core/api.py"], "full"),
        (["chap_core/api.py", "README.md"], "full"),
        (["docs/index.md", "chap_core/api.py"], "full"),
        (["chap_core/api.py", "docs/index.md"], "full"),
        (["README.md", "docs/index.md"], "docs"),
        (["docs/index.md", "README.md"], "docs"),
    ],
)
def test_scope_for_changed_files(classify_script, tmp_path, changed_files, expected):
    assert run_classify(classify_script, tmp_path, changed_files) == expected


@needs_bash
def test_push_events_always_run_the_full_suite(classify_script, tmp_path):
    """Whatever a pull request skipped is still verified once it lands."""
    assert run_classify(classify_script, tmp_path, ["README.md"], event_name="push") == "full"


@needs_bash
def test_skip_ci_label_overrides_a_code_change(classify_script, tmp_path):
    assert run_classify(classify_script, tmp_path, ["chap_core/api.py"], skip_label="true") == "inert"


def test_gated_jobs_keep_the_names_required_by_branch_protection(repo_root):
    """The ruleset requires these check names; renaming a job would block merges."""
    workflows = {
        "ci-lint-code-style.yml": ("check", "needs.changes.outputs.scope != 'inert'"),
        "ci-test-python-install.yml": ("test", "needs.changes.outputs.scope != 'inert'"),
        "ci-test-external-models.yml": ("test-all", "needs.changes.outputs.scope == 'full'"),
    }
    for filename, (job_name, condition) in workflows.items():
        workflow = yaml.safe_load((repo_root / ".github" / "workflows" / filename).read_text())
        assert job_name in workflow["jobs"], f"{filename} lost the required job {job_name}"
        job = workflow["jobs"][job_name]
        assert job["if"] == condition
        assert job["needs"] == "changes"
