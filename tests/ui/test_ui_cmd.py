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
        ui_cmd(port=9999, workdir=tmp_path / "work", open_browser=False)
    cmd, env = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py"
    assert cmd[cmd.index("--server.port") + 1] == "9999"
    assert cmd[cmd.index("--server.headless") + 1] == "true"
    assert env["CHAP_UI_WORKDIR"] == str((tmp_path / "work").resolve())
