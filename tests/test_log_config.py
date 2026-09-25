import logging

from chap_core.log_config import STATUS_LOGGER_NAME, get_status_logger, initialize_logging


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


def test_get_status_logger_reenables_a_logger_disabled_by_logging_config():
    logging.getLogger(STATUS_LOGGER_NAME).disabled = True

    assert get_status_logger().disabled is False
