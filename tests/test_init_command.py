"""Tests for the chap init command."""

import subprocess
from pathlib import Path

import pytest

from chap_core.cli_endpoints.init import init
from chap_core.util import uv_available


def test_init_creates_expected_files(tmp_path, monkeypatch):
    """Test that init creates all expected files."""
    monkeypatch.chdir(tmp_path)

    init("test_model")

    model_dir = tmp_path / "test_model"
    assert model_dir.exists()
    assert (model_dir / "MLproject").exists()
    assert (model_dir / "pyproject.toml").exists()
    assert (model_dir / "main.py").exists()
    assert (model_dir / "README.md").exists()
    assert (model_dir / "isolated_run.py").exists()
    assert (model_dir / "input" / "trainData.csv").exists()
    assert (model_dir / "input" / "futureClimateData.csv").exists()
    assert (model_dir / "output").exists()


def test_init_mlproject_has_correct_content(tmp_path, monkeypatch):
    """Test that MLproject file has correct uv_env configuration."""
    monkeypatch.chdir(tmp_path)

    init("my_model")

    mlproject_content = (tmp_path / "my_model" / "MLproject").read_text()
    assert "name: my_model" in mlproject_content
    assert "uv_env: pyproject.toml" in mlproject_content
    assert "train:" in mlproject_content
    assert "predict:" in mlproject_content


def test_init_pyproject_has_correct_content(tmp_path, monkeypatch):
    """Test that pyproject.toml has correct dependencies."""
    monkeypatch.chdir(tmp_path)

    init("my_model")

    pyproject_content = (tmp_path / "my_model" / "pyproject.toml").read_text()
    assert 'name = "my_model"' in pyproject_content
    assert "pandas" in pyproject_content
    assert "scikit-learn" in pyproject_content
    assert "cyclopts" in pyproject_content


def test_init_main_py_has_cyclopts_commands(tmp_path, monkeypatch):
    """Test that main.py has train and predict commands using cyclopts."""
    monkeypatch.chdir(tmp_path)

    init("my_model")

    main_py_content = (tmp_path / "my_model" / "main.py").read_text()
    assert "from cyclopts import App" in main_py_content
    assert "@app.command()" in main_py_content
    assert "def train(" in main_py_content
    assert "def predict(" in main_py_content


def test_init_fails_if_directory_exists(tmp_path, monkeypatch):
    """Test that init fails if directory already exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing_model").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        init("existing_model")


def test_init_via_cli(tmp_path):
    """Test that chap init works via CLI."""
    result = subprocess.run(
        ["chap", "init", "cli_test_model"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert result.returncode == 0, f"init failed: {result.stderr.decode()}"

    model_dir = tmp_path / "cli_test_model"
    assert (model_dir / "MLproject").exists()
    assert (model_dir / "pyproject.toml").exists()
    assert (model_dir / "main.py").exists()
    assert (model_dir / "README.md").exists()
    assert (model_dir / "isolated_run.py").exists()
    assert (model_dir / "input" / "trainData.csv").exists()
    assert (model_dir / "input" / "futureClimateData.csv").exists()


@pytest.mark.slow
@pytest.mark.skipif(not uv_available(), reason="Requires uv")
def test_init_model_runs_through_evaluate(tmp_path):
    """Test that a fresh init model can be evaluated end-to-end."""
    # 1. Init the model via CLI
    result = subprocess.run(
        ["chap", "init", "eval_test_model"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert result.returncode == 0, f"init failed: {result.stderr.decode()}"

    model_dir = tmp_path / "eval_test_model"

    # 2. Run uv sync in the model directory
    result = subprocess.run(
        ["uv", "sync"],
        cwd=model_dir,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, f"uv sync failed: {result.stderr.decode()}"

    # 3. Run chap evaluate2 with example data
    example_data = Path("example_data/laos_subset.csv").resolve()
    output_file = tmp_path / "eval.nc"

    result = subprocess.run(
        [
            "chap",
            "evaluate2",
            "--model-name",
            str(model_dir),
            "--dataset-csv",
            str(example_data),
            "--output-file",
            str(output_file),
            "--backtest-params.n-splits",
            "2",
            "--backtest-params.n-periods",
            "1",
        ],
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, f"evaluate2 failed: {result.stderr.decode()}"
    assert output_file.exists()
