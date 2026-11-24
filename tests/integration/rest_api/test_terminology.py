from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.terminology import load_terminology

client = TestClient(app)


def test_list_terminology_returns_every_term_and_the_source():
    terminology = load_terminology()
    response = client.get("/v1/terminology")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == {
        "repository": "dhis2-chap/user-stories",
        "path": "terminology.yaml",
        "commit": terminology.commit,
        "date": terminology.date,
    }
    assert [term["id"] for term in body["terms"]] == [term["id"] for term in terminology.terms]
    for served, term in zip(body["terms"], terminology.terms, strict=True):
        assert served["definition"] == term["definition"]
        assert served["seeAlso"] == term.get("see_also", [])


def test_read_term_finds_a_term_by_name_and_flags_avoided_names():
    term = next(term for term in load_terminology().terms if "avoid" in term)
    response = client.get(f"/v1/terminology/{term['term']}s")
    assert response.status_code == 200, response.text
    assert response.json()["term"]["id"] == term["id"]
    assert response.json()["nameIsAvoided"] is False

    response = client.get(f"/v1/terminology/{term['avoid'][0]}")
    assert response.status_code == 200, response.text
    assert response.json()["term"]["id"] == term["id"]
    assert response.json()["nameIsAvoided"] is True


def test_read_term_of_unknown_name_is_404():
    response = client.get("/v1/terminology/no such term")
    assert response.status_code == 404
    assert "no such term" in response.json()["detail"]
    assert "/v1/terminology" in response.json()["detail"]
