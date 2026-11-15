"""Request-scoped translation dependency for endpoints returning localized text."""

from gettext import NullTranslations

from fastapi import Header, Response

from chap_core.i18n import load_translations, select_language


def get_translations(response: Response, accept_language: str = Header(default="")) -> NullTranslations:
    translations = load_translations()
    language = select_language(accept_language, translations)
    response.headers["Content-Language"] = language
    response.headers["Vary"] = "Accept-Language"
    return translations[language]
