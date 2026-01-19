"""Tests for the approved templates module."""

from unittest.mock import MagicMock, patch

import pytest

from chap_core.models.approved_templates import (
    ApprovedTemplate,
    ApprovedTemplatesCache,
    fetch_approved_templates_from_url,
    get_approved_templates,
    get_git_ref,
    is_approved,
    clear_cache,
)


class TestApprovedTemplate:
    def test_model_validation(self):
        template = ApprovedTemplate(
            url="https://github.com/dhis2-chap/chap_auto_ewars",
            versions={"stable": "@abc123", "nightly": "@main"},
        )
        assert template.url == "https://github.com/dhis2-chap/chap_auto_ewars"
        assert template.versions["stable"] == "@abc123"


class TestApprovedTemplatesCache:
    def test_cache_initially_empty(self):
        cache = ApprovedTemplatesCache()
        assert cache.get() is None

    def test_cache_set_and_get(self):
        cache = ApprovedTemplatesCache()
        templates = [ApprovedTemplate(url="https://example.com", versions={"v1": "@commit"})]
        cache.set(templates)
        assert cache.get() == templates

    def test_cache_clear(self):
        cache = ApprovedTemplatesCache()
        templates = [ApprovedTemplate(url="https://example.com", versions={"v1": "@commit"})]
        cache.set(templates)
        cache.clear()
        assert cache.get() is None


class TestIsApproved:
    @pytest.fixture
    def approved_list(self):
        return [
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/chap_auto_ewars",
                versions={"stable": "@abc123", "nightly": "@main"},
            ),
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/ewars_template",
                versions={"v3": "@def456"},
            ),
        ]

    def test_approved_url_and_version(self, approved_list):
        assert is_approved("https://github.com/dhis2-chap/chap_auto_ewars", "stable", approved_list)

    def test_approved_different_version(self, approved_list):
        assert is_approved("https://github.com/dhis2-chap/chap_auto_ewars", "nightly", approved_list)

    def test_not_approved_wrong_url(self, approved_list):
        assert not is_approved("https://github.com/other/repo", "stable", approved_list)

    def test_not_approved_wrong_version(self, approved_list):
        assert not is_approved("https://github.com/dhis2-chap/chap_auto_ewars", "unknown", approved_list)

    def test_empty_approved_list(self):
        assert not is_approved("https://example.com", "v1", [])


class TestGetGitRef:
    @pytest.fixture
    def approved_list(self):
        return [
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/chap_auto_ewars",
                versions={"stable": "@abc123", "nightly": "@main"},
            ),
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/ewars_template",
                versions={"v3": "@def456"},
            ),
        ]

    def test_get_git_ref_for_approved_version(self, approved_list):
        ref = get_git_ref("https://github.com/dhis2-chap/chap_auto_ewars", "stable", approved_list)
        assert ref == "@abc123"

    def test_get_git_ref_for_different_version(self, approved_list):
        ref = get_git_ref("https://github.com/dhis2-chap/chap_auto_ewars", "nightly", approved_list)
        assert ref == "@main"

    def test_get_git_ref_returns_none_for_wrong_url(self, approved_list):
        ref = get_git_ref("https://github.com/other/repo", "stable", approved_list)
        assert ref is None

    def test_get_git_ref_returns_none_for_wrong_version(self, approved_list):
        ref = get_git_ref("https://github.com/dhis2-chap/chap_auto_ewars", "unknown", approved_list)
        assert ref is None


class TestFetchApprovedTemplatesFromUrl:
    def test_fetch_and_parse(self):
        yaml_content = """
- url: https://github.com/dhis2-chap/chap_auto_ewars
  versions:
    stable: "@abc123"
"""
        mock_response = MagicMock()
        mock_response.text = yaml_content

        with patch("chap_core.models.approved_templates.requests.get") as mock_get:
            mock_get.return_value = mock_response
            templates = fetch_approved_templates_from_url("https://example.com/whitelist.yaml")

        assert len(templates) == 1
        assert templates[0].url == "https://github.com/dhis2-chap/chap_auto_ewars"
        assert templates[0].versions["stable"] == "@abc123"

    def test_fetch_empty_file(self):
        mock_response = MagicMock()
        mock_response.text = ""

        with patch("chap_core.models.approved_templates.requests.get") as mock_get:
            mock_get.return_value = mock_response
            templates = fetch_approved_templates_from_url("https://example.com/whitelist.yaml")

        assert templates == []


class TestGetApprovedTemplates:
    def setup_method(self):
        clear_cache()

    def test_caches_result(self):
        yaml_content = """
- url: https://github.com/example/repo
  versions:
    v1: "@commit123"
"""
        mock_response = MagicMock()
        mock_response.text = yaml_content

        with patch("chap_core.models.approved_templates.requests.get") as mock_get:
            mock_get.return_value = mock_response
            with patch("chap_core.models.approved_templates.load_repo_urls") as mock_urls:
                mock_urls.return_value = ["https://example.com/whitelist.yaml"]

                templates1 = get_approved_templates()
                templates2 = get_approved_templates()

        assert templates1 == templates2
        assert mock_get.call_count == 1

    def test_returns_merged_results(self):
        yaml1 = """
- url: https://github.com/example/repo1
  versions:
    v1: "@commit1"
"""
        yaml2 = """
- url: https://github.com/example/repo2
  versions:
    v1: "@commit2"
"""
        mock_response1 = MagicMock()
        mock_response1.text = yaml1
        mock_response2 = MagicMock()
        mock_response2.text = yaml2

        with patch("chap_core.models.approved_templates.requests.get") as mock_get:
            mock_get.side_effect = [mock_response1, mock_response2]
            with patch("chap_core.models.approved_templates.load_repo_urls") as mock_urls:
                mock_urls.return_value = [
                    "https://example.com/whitelist1.yaml",
                    "https://example.com/whitelist2.yaml",
                ]

                templates = get_approved_templates()

        assert len(templates) == 2
        urls = {t.url for t in templates}
        assert "https://github.com/example/repo1" in urls
        assert "https://github.com/example/repo2" in urls
