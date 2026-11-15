"""Gettext catalogs for API text. Source strings stay English until a response is built."""

from functools import cache
from gettext import GNUTranslations, NullTranslations
from io import BytesIO
from pathlib import Path

from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po

LOCALES_DIR = Path(__file__).parent / "locales"


def N_(message: str) -> str:
    """Mark a source string for extraction without translating it at import time."""
    return message


@cache
def load_translations() -> dict[str, NullTranslations]:
    """Load shipped PO files once, compiling in memory so wheels and Docker work alike."""
    translations: dict[str, NullTranslations] = {"en": NullTranslations()}
    for path in sorted(LOCALES_DIR.glob("*.po")):
        with path.open("rb") as source:
            catalog = read_po(source)
        compiled = BytesIO()
        write_mo(compiled, catalog)
        compiled.seek(0)
        translations[path.stem.replace("_", "-").lower()] = GNUTranslations(compiled)
    return translations


def select_language(accept_language: str, available: dict[str, NullTranslations]) -> str:
    """Honor language preferences and quality weights, with English as the fallback."""
    preferences = []
    for item in accept_language.split(","):
        language, _, parameter = item.strip().partition(";")
        quality = 1.0
        if parameter:
            key, _, value = parameter.strip().partition("=")
            if key.lower() != "q":
                continue
            try:
                quality = float(value)
            except ValueError:
                continue
        if 0 < quality <= 1:
            preferences.append((language.strip().replace("_", "-").lower(), quality))
    for language, _ in sorted(preferences, key=lambda preference: preference[1], reverse=True):
        if language == "*":
            return "en"
        while language:
            if language in available:
                return language
            language = language.rpartition("-")[0]
    return "en"
