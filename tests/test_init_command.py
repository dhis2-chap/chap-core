"""Tests for the chap init command."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from chap_core.cli_endpoints.init import init, _init_renv, _check_r_and_renv_available
from chap_core.util import uv_available


def _mock_renv_init(target_dir: Path):
    """Simulate what renv::init() creates for testing purposes."""
    renv_dir = target_dir / "renv"
    renv_dir.mkdir(exist_ok=True)
    (target_dir / ".Rprofile").write_text('source("renv/activate.R")\n')
    (target_dir / "renv.lock").write_text('{"R": {}, "Packages": {}}\n')
    (renv_dir / "activate.R").write_text("# renv activate script\n")


def _r_and_renv_available():
    """Check if R and renv are available for testing."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "if (!requireNamespace('renv', quietly=TRUE)) quit(status=1)"],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


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


def test_init_r_creates_expected_files(tmp_path, monkeypatch):
    """Test that init with language=r creates all expected R files."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "chap_core.cli_endpoints.init._init_renv",
        side_effect=lambda d: _mock_renv_init(d),
    ):
        init("test_r_model", language="r")

    model_dir = tmp_path / "test_r_model"
    assert model_dir.exists()
    assert (model_dir / "MLproject").exists()
    assert (model_dir / "main.R").exists()
    assert (model_dir / "README.md").exists()
    assert (model_dir / "isolated_run.R").exists()
    assert (model_dir / ".Rprofile").exists()
    assert (model_dir / "renv.lock").exists()
    assert (model_dir / "renv" / "activate.R").exists()
    assert (model_dir / "input" / "trainData.csv").exists()
    assert (model_dir / "input" / "futureClimateData.csv").exists()
    assert (model_dir / "output").exists()


def test_init_r_mlproject_has_correct_content(tmp_path, monkeypatch):
    """Test that R MLproject file has correct renv_env configuration."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "chap_core.cli_endpoints.init._init_renv",
        side_effect=lambda d: _mock_renv_init(d),
    ):
        init("my_r_model", language="r")

    mlproject_content = (tmp_path / "my_r_model" / "MLproject").read_text()
    assert "name: my_r_model" in mlproject_content
    assert "renv_env: renv.lock" in mlproject_content
    assert "Rscript main.R train" in mlproject_content
    assert "Rscript main.R predict" in mlproject_content


def test_init_r_main_has_train_and_predict(tmp_path, monkeypatch):
    """Test that main.R has train and predict functions using optparse."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "chap_core.cli_endpoints.init._init_renv",
        side_effect=lambda d: _mock_renv_init(d),
    ):
        init("my_r_model", language="r")

    main_r_content = (tmp_path / "my_r_model" / "main.R").read_text()
    assert "library(optparse)" in main_r_content
    assert "train <- function" in main_r_content
    assert "predict_model <- function" in main_r_content
    assert "lm(" in main_r_content
    assert "make_option" in main_r_content


def test_init_renv_calls_correct_commands(tmp_path):
    """Test that _init_renv calls the correct renv commands."""
    with patch("chap_core.cli_endpoints.init._check_r_and_renv_available"):
        with patch("chap_core.cli_endpoints.init.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _init_renv(tmp_path)

    # Verify the correct commands were called
    assert mock_run.call_count == 3

    calls = mock_run.call_args_list
    # First call: renv::init()
    assert calls[0][0][0] == ["Rscript", "-e", "renv::init()"]
    assert calls[0][1]["cwd"] == tmp_path

    # Second call: renv::install('optparse')
    assert calls[1][0][0] == ["Rscript", "-e", "renv::install('optparse')"]
    assert calls[1][1]["cwd"] == tmp_path

    # Third call: renv::snapshot(prompt=FALSE)
    assert calls[2][0][0] == ["Rscript", "-e", "renv::snapshot(prompt=FALSE)"]
    assert calls[2][1]["cwd"] == tmp_path


def test_check_r_and_renv_raises_when_r_not_installed():
    """Test that _check_r_and_renv_available raises when R is not installed."""
    with patch(
        "chap_core.cli_endpoints.init.subprocess.run",
        side_effect=FileNotFoundError("Rscript not found"),
    ):
        with pytest.raises(RuntimeError, match="R is not installed"):
            _check_r_and_renv_available()


def test_check_r_and_renv_raises_when_renv_not_installed():
    """Test that _check_r_and_renv_available raises when renv is not installed."""

    def mock_run(cmd, **kwargs):
        if "renv" in str(cmd):
            return MagicMock(returncode=1)
        return MagicMock(returncode=0)

    with patch("chap_core.cli_endpoints.init.subprocess.run", side_effect=mock_run):
        with pytest.raises(RuntimeError, match="renv package is not installed"):
            _check_r_and_renv_available()


@pytest.mark.slow
@pytest.mark.skipif(not _r_and_renv_available(), reason="Requires R and renv")
def test_init_r_via_cli(tmp_path):
    """Test that chap init --language r works via CLI."""
    result = subprocess.run(
        ["chap", "init", "cli_r_test_model", "--language", "r"],
        cwd=tmp_path,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, f"init failed: {result.stderr.decode()}"

    model_dir = tmp_path / "cli_r_test_model"
    assert (model_dir / "MLproject").exists()
    assert (model_dir / "main.R").exists()
    assert (model_dir / "renv.lock").exists()
    assert (model_dir / "renv" / "activate.R").exists()


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

    # 2. Run chap evaluate2 with example data (uv run handles dependencies automatically)
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
