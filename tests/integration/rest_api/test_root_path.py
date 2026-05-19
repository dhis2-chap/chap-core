import importlib

import chap_core.rest_api.app as app_module


class TestRootPath:
    def test_root_path_defaults_to_empty(self, monkeypatch):
        monkeypatch.delenv("CHAP_ROOT_PATH", raising=False)
        reloaded = importlib.reload(app_module)
        try:
            assert reloaded.app.root_path == ""
        finally:
            importlib.reload(app_module)

    def test_root_path_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("CHAP_ROOT_PATH", "/master")
        reloaded = importlib.reload(app_module)
        try:
            assert reloaded.app.root_path == "/master"
        finally:
            monkeypatch.delenv("CHAP_ROOT_PATH", raising=False)
            importlib.reload(app_module)
