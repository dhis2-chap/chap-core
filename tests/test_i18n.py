from gettext import NullTranslations

import pytest

from chap_core import i18n


@pytest.mark.parametrize(
    "accept_language,language",
    [
        ("fr-CA, en;q=0.5", "fr"),
        ("fr;q=0.4, en;q=0.8", "en"),
        ("de, fr;q=0.8", "fr"),
        ("fr;q=0, en", "en"),
        ("fr;q=invalid", "en"),
        ("", "en"),
    ],
)
def test_select_language(accept_language, language):
    available = {"en": NullTranslations(), "fr": NullTranslations()}
    assert i18n.select_language(accept_language, available) == language


def test_empty_text_is_not_translated_to_the_catalog_header(french_catalog):
    token = i18n.current_translations.set(i18n.load_translations()["fr"])
    try:
        assert i18n.gettext("") == ""
        assert i18n.gettext("Global median") == french_catalog["Global median"]
        assert i18n.gettext("RMSE") == "RMSE"
    finally:
        i18n.current_translations.reset(token)


def test_broken_catalog_falls_back_to_english(tmp_path, monkeypatch):
    (tmp_path / "fr.po").write_text('msgid "Map"\nmsgstr "Carte"\nnot po syntax\n')
    monkeypatch.setattr(i18n, "LOCALES_DIR", tmp_path)
    i18n.load_translations.cache_clear()
    try:
        assert set(i18n.load_translations()) == {"en"}
    finally:
        i18n.load_translations.cache_clear()
