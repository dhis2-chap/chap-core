import requests

from chap_core.external.github import resolve_commit_sha


def test_resolve_commit_sha_returns_pinned_sha_without_lookup(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a full sha must not need a lookup")

    monkeypatch.setattr("chap_core.external.github.requests.get", fail)
    commit_sha = "0c41b1d9bd187521e62c58d581e6f5bd5127f7b5"

    assert resolve_commit_sha(f"https://github.com/example/test_model@{commit_sha}") == commit_sha


def test_resolve_commit_sha_returns_uppercase_sha_without_lookup(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a full sha must not need a lookup")

    monkeypatch.setattr("chap_core.external.github.requests.get", fail)
    commit_sha = "0C41B1D9BD187521E62C58D581E6F5BD5127F7B5"

    assert resolve_commit_sha(f"https://github.com/example/test_model@{commit_sha}") == commit_sha.lower()


def test_resolve_commit_sha_returns_abbreviated_sha_without_lookup(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an abbreviated sha must not need a lookup")

    monkeypatch.setattr("chap_core.external.github.requests.get", fail)
    commit_sha = "0c41b1d"

    assert resolve_commit_sha(f"https://github.com/example/test_model@{commit_sha}") == commit_sha


def test_resolve_commit_sha_returns_none_when_lookup_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr("chap_core.external.github.requests.get", fail)

    assert resolve_commit_sha("https://github.com/example/test_model@main") is None
