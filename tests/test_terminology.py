import re
from datetime import datetime

import pytest

from chap_core.terminology import COPY_PATH, TermMatch, load_terminology, lookup_term, split_copy, validate_terms
from scripts import sync_terminology

TERMS = load_terminology().terms
VALID_TERM = {"id": "a-term", "term": "A term", "group": "Models", "status": "current", "definition": "Text."}


def test_bundled_copy_is_valid_and_names_its_source():
    terminology = load_terminology()
    assert validate_terms(terminology.terms) == []
    assert re.fullmatch(r"[0-9a-f]{40}", terminology.commit)
    assert datetime.fromisoformat(terminology.date)


def find(name: str) -> TermMatch:
    match = lookup_term(name)
    assert match is not None, name
    return match


@pytest.mark.parametrize("term", TERMS, ids=lambda term: term["id"])
def test_every_name_of_a_term_leads_to_it(term):
    used = [term["id"], term["term"], *term.get("aliases", [])]
    for name in [*used, *(f"the {name}s" for name in used), term["term"].upper().replace(" ", "_")]:
        assert find(name).term is term, name
    for name in used:
        assert not find(name).avoided, name
    for name in term.get("avoid", []):
        assert find(name).term is term and find(name).avoided, name


def test_lookup_examples():
    assert find("baseline_models").term["id"] == "baseline-model"
    assert find("  The  chap/core ").term["id"] == "chap-core"
    assert lookup_term("no such term") is None
    assert lookup_term("") is None


@pytest.mark.parametrize(
    "terms,error",
    [
        ("text", "non-empty list"),
        ([{**VALID_TERM, "id": "Not Kebab"}], "kebab-case"),
        ([VALID_TERM, VALID_TERM], "duplicate id"),
        ([{k: v for k, v in VALID_TERM.items() if k != "definition"}], "missing `definition`"),
        ([{**VALID_TERM, "extra": 1}], "unknown field `extra`"),
        ([{**VALID_TERM, "group": "Other"}], "`group` must be one of"),
        ([{**VALID_TERM, "status": "draft"}], "`status` must be one of"),
        ([VALID_TERM, {**VALID_TERM, "id": "b-term", "term": "B term", "avoid": ["a_TERM"]}], "already a name of"),
        ([{**VALID_TERM, "see_also": ["missing"]}], "not a term id"),
    ],
)
def test_validation_reports_broken_terms(terms, error):
    assert validate_terms([VALID_TERM]) == []
    assert any(error in message for message in validate_terms(terms))


def test_sync_writes_a_changed_source_but_never_an_invalid_one(tmp_path, monkeypatch):
    copy = tmp_path / "terminology.yaml"
    source = split_copy(COPY_PATH.read_text(encoding="utf-8"))[1]
    monkeypatch.setattr(sync_terminology, "COPY_PATH", copy)
    monkeypatch.setattr(sync_terminology, "fetch_source", lambda: ("a" * 40, "2026-01-01T00:00:00Z", source))

    assert sync_terminology.main(["--check"]) == 1
    assert not copy.exists()
    assert sync_terminology.main([]) == 0
    header = {"source-commit": "a" * 40, "source-date": "2026-01-01T00:00:00Z"}
    assert split_copy(copy.read_text(encoding="utf-8")) == (header, source)
    assert sync_terminology.main(["--check"]) == 0

    written = copy.read_bytes()
    monkeypatch.setattr(sync_terminology, "fetch_source", lambda: ("b" * 40, "2026-01-02T00:00:00Z", source))
    assert sync_terminology.main([]) == 0
    assert copy.read_bytes() == written, "an unchanged source must not rewrite the copy"

    monkeypatch.setattr(sync_terminology, "fetch_source", lambda: ("c" * 40, "2026-01-03T00:00:00Z", "- id: Broken\n"))
    assert sync_terminology.main(["--check"]) == 1
    assert sync_terminology.main([]) == 2
    assert copy.read_bytes() == written
