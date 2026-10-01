from pathlib import Path

import pytest

from chap_core.cli_endpoints import ui
from chap_core.cli_endpoints.ui import INSTALL_HINT, ui_cmd


def test_ui_cmd_prints_install_hint_without_streamlit(monkeypatch, capsys):
    monkeypatch.setattr(ui.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        ui_cmd()
    assert exc.value.code == 1
    assert INSTALL_HINT in capsys.readouterr().out


def test_ui_cmd_starts_streamlit_with_workdir(monkeypatch, tmp_path):
    calls = []

    def fake_call(cmd, env):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(ui.subprocess, "call", fake_call)
    with pytest.raises(SystemExit):
        ui_cmd(port=9999, runs_dir=tmp_path / "runs", open_browser=False)
    cmd, env = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py"
    assert cmd[cmd.index("--server.port") + 1] == "9999"
    assert cmd[cmd.index("--server.headless") + 1] == "true"
    assert env["CHAP_RUNS_DIR"] == str((tmp_path / "runs").resolve())
    assert not (tmp_path / "runs").exists()
    assert Path(cmd[cmd.index("--theme.base") + 1]).is_file()


def test_ui_cmd_passes_folders_models_and_marketplace_to_the_ui(monkeypatch, tmp_path):
    calls = []

    def fake_call(cmd, env):
        calls.append(env)
        return 0

    monkeypatch.setattr(ui.subprocess, "call", fake_call)
    with pytest.raises(SystemExit):
        ui_cmd(
            runs_dir=tmp_path / "runs",
            uploads_dir=tmp_path / "data",
            models=tmp_path / "team.yaml",
            chaps_project=tmp_path / "deploy",
            registry_url="https://example.org/registry",
        )
    env = calls[0]
    assert env["CHAP_UPLOADS_DIR"] == str((tmp_path / "data").resolve())
    assert env["CHAP_MODELS_FILE"] == str((tmp_path / "team.yaml").resolve())
    assert env["CHAPS_PROJECT_DIR"] == str((tmp_path / "deploy").resolve())
    assert env["CHAP_MARKETPLACE_URL"] == "https://example.org/registry"


def test_install_hint_shows_how_to_run_with_uvx():
    assert "uvx --from 'chap-core[ui]' chap ui" in INSTALL_HINT


@pytest.mark.parametrize(
    "given",
    ["https://example.org/registry", "https://example.org/registry/", "https://example.org/registry/registry.yaml"],
)
def test_registry_url_is_passed_as_the_marketplace_base_url(monkeypatch, given):
    calls = []

    def fake_call(cmd, env):
        calls.append(env)
        return 0

    monkeypatch.setattr(ui.subprocess, "call", fake_call)
    with pytest.raises(SystemExit):
        ui_cmd(registry_url=given)
    assert calls[0]["CHAP_MARKETPLACE_URL"] == "https://example.org/registry"


def test_ui_listens_only_on_this_machine_unless_told_otherwise(monkeypatch):
    calls = []

    def fake_call(cmd, env):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(ui.subprocess, "call", fake_call)
    with pytest.raises(SystemExit):
        ui_cmd()
    with pytest.raises(SystemExit):
        ui_cmd(host="0.0.0.0")
    assert [cmd[cmd.index("--server.address") + 1] for cmd in calls] == ["127.0.0.1", "0.0.0.0"]
