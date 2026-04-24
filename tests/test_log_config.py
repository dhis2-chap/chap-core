import logging

from chap_core.log_config import initialize_logging


def test_initialize_logging_defaults_to_info_when_env_unset(monkeypatch):
    monkeypatch.delenv("CHAP_DEBUG", raising=False)
    logging.getLogger().setLevel(logging.WARNING)

    initialize_logging()

    assert logging.getLogger().level == logging.INFO


def test_initialize_logging_respects_chap_debug_env(monkeypatch):
    monkeypatch.setenv("CHAP_DEBUG", "true")
    logging.getLogger().setLevel(logging.WARNING)

    initialize_logging()

    assert logging.getLogger().level == logging.DEBUG
