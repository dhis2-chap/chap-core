"""The Chap team's terminology: the terms, a check of the file's format, and a lookup by name.

``terminology.yaml`` in this package is a copy of ``terminology.yaml`` in dhis2-chap/user-stories,
kept up to date by ``scripts/sync_terminology.py``. It is never edited here.

This module uses only the standard library and PyYAML, so the sync script can import it
without the rest of chap-core's dependencies.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

SOURCE_REPOSITORY = "dhis2-chap/user-stories"
SOURCE_PATH = "terminology.yaml"
COPY_PATH = Path(__file__).parent / "terminology.yaml"

GROUPS = ["Products", "Models", "Evaluation", "Model Marketplace"]
STATUSES = {"current", "proposed"}
REQUIRED = {"id", "term", "group", "status", "definition"}
OPTIONAL = {"note", "details", "aliases", "avoid", "see_also"}
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
HEADER_FIELD_RE = re.compile(r"^# (source-commit|source-date): (\S+)$", re.MULTILINE)
ARTICLES = ("the ", "a ", "an ")

Term = dict[str, Any]


@dataclass(frozen=True)
class Terminology:
    terms: list[Term]
    commit: str
    date: str


@dataclass(frozen=True)
class TermMatch:
    term: Term
    avoided: bool
    """Whether the name asked for is one of the term's ``avoid`` forms."""


def split_copy(text: str) -> tuple[dict[str, str], str]:
    """The header fields (``source-commit``, ``source-date``) and the source content of a synced copy."""
    header, _, content = text.partition("\n\n")
    return dict(HEADER_FIELD_RE.findall(header)), content


@cache
def load_terminology() -> Terminology:
    """The bundled copy of the terminology, read once."""
    header, content = split_copy(COPY_PATH.read_text(encoding="utf-8"))
    return Terminology(terms=yaml.safe_load(content), commit=header["source-commit"], date=header["source-date"])


def is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def spelling(name: str) -> str:
    """A name with `-`, `_` and `/` as spaces and whitespace collapsed, in its own case."""
    return " ".join(re.sub(r"[-_/]+", " ", name).split())


def name_key(name: str) -> str:
    """A name as lower-case words, so 'chap-core', 'chap_core' and 'Chap core' are one name."""
    return spelling(name).lower()


def validate_terms(terms: object) -> list[str]:
    """The errors in a parsed terminology file. An empty list means it is valid.

    The rules are those of ``check_terminology.py`` in dhis2-chap/user-stories.
    """
    if not isinstance(terms, list) or not terms:
        return ["must be a non-empty list of terms"]
    errors: list[str] = []
    ids: set[str] = set()
    names: dict[str, str] = {}  # name key -> id of the term that has it
    for i, entry in enumerate(terms):
        if not isinstance(entry, dict):
            errors.append(f"term #{i + 1}: is not a mapping")
            continue
        term_id = entry.get("id")
        where = term_id if isinstance(term_id, str) and term_id else f"term #{i + 1}"
        keys = set(entry)
        errors += [f"{where}: missing `{key}`" for key in sorted(REQUIRED - keys)]
        errors += [f"{where}: unknown field `{key}`" for key in sorted(keys - REQUIRED - OPTIONAL, key=str)]

        if term_id is not None:
            if not (isinstance(term_id, str) and ID_RE.match(term_id)):
                errors.append(f"{where}: `id` must be kebab-case, got {term_id!r}")
            elif term_id in ids:
                errors.append(f"{where}: duplicate id")
            else:
                ids.add(term_id)

        for key in ("term", "definition", "note", "details"):
            if key in entry and not is_text(entry[key]):
                errors.append(f"{where}: `{key}` must be non-empty text")
        if "group" in entry and entry["group"] not in GROUPS:
            errors.append(f"{where}: `group` must be one of {GROUPS}, got {entry['group']!r}")
        if "status" in entry and entry["status"] not in STATUSES:
            errors.append(f"{where}: `status` must be one of {sorted(STATUSES)}, got {entry['status']!r}")

        entry_names = [entry["term"]] if is_text(entry.get("term")) else []
        for key in ("aliases", "avoid"):
            value = entry.get(key)
            if value is None:
                continue
            if not isinstance(value, list) or not value or not all(is_text(x) for x in value):
                errors.append(f"{where}: `{key}` must be a non-empty list of names")
            else:
                entry_names += value
        # One name leads to one term, so a lookup has one answer.
        # A term may list a spelling of its own name ("CHAP" under Chap), but not the same one twice.
        if len(set(entry_names)) != len(entry_names):
            errors.append(f"{where}: a name is listed twice in `term`, `aliases` and `avoid`")
        for name in {name_key(name) for name in entry_names}:
            if name in names:
                errors.append(f"{where}: the name {name!r} is already a name of `{names[name]}`")
            else:
                names[name] = where

    for entry in terms:
        if not isinstance(entry, dict) or "see_also" not in entry:
            continue
        where = entry.get("id") or "?"
        see_also = entry["see_also"]
        if not isinstance(see_also, list) or not see_also or not all(isinstance(x, str) for x in see_also):
            errors.append(f"{where}: `see_also` must be a non-empty list of term ids")
            continue
        for other in see_also:
            if other == entry.get("id"):
                errors.append(f"{where}: `see_also` points to the term itself")
            elif other not in ids:
                errors.append(f"{where}: `see_also` names `{other}`, which is not a term id")
        if len(set(see_also)) != len(see_also):
            errors.append(f"{where}: `see_also` lists a term twice")
    return errors


@cache
def name_index() -> dict[str, tuple[Term, bool]]:
    """Name key -> the term with that name, and whether the name is only an ``avoid`` form of it."""
    terms = load_terminology().terms
    index: dict[str, tuple[Term, bool]] = {}
    for term in terms:
        for name in [term["term"], *term.get("aliases", [])]:
            index[name_key(name)] = (term, False)
    for term in terms:
        for name in term.get("avoid", []):
            index.setdefault(name_key(name), (term, True))
        index.setdefault(name_key(term["id"]), (term, False))
    return index


def lookup_term(name: str) -> TermMatch | None:
    """The term that goes by a name: its ``id``, ``term``, an alias or an avoided form.

    Case, `-`, `_`, `/` and extra whitespace are ignored, as are a leading "the", "a" or "an"
    and a plural "s". A name that the term also lists in another case under ``avoid``
    ("CHAP" for Chap) counts as avoided only when written in that case.
    """
    candidates = [spelling(name)]
    for article in ARTICLES:
        if candidates[0].lower().startswith(article):
            candidates.append(candidates[0][len(article) :])
    candidates += [candidate[:-1] for candidate in candidates if candidate.lower().endswith("s")]
    index = name_index()
    for candidate in candidates:
        if candidate.lower() in index:
            term, only_avoided = index[candidate.lower()]
            avoided = only_avoided or candidate in {spelling(avoid) for avoid in term.get("avoid", [])}
            return TermMatch(term=term, avoided=avoided)
    return None
