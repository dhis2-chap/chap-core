"""
Module for managing approved model templates via remote whitelists.

This module provides functionality to:
1. Load whitelist URLs from local config
2. Fetch approved model templates from remote YAML files
3. Validate model template requests against the whitelist
4. Cache whitelist data to avoid repeated fetches
"""

import logging
import time
from pathlib import Path

import requests
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CACHE_TTL = 300  # 5 minutes


class ApprovedTemplate(BaseModel):
    url: str
    versions: dict[str, str]


class ApprovedTemplatesCache:
    def __init__(self):
        self._data: list[ApprovedTemplate] | None = None
        self._timestamp: float = 0

    def get(self) -> list[ApprovedTemplate] | None:
        if self._data is not None and time.time() - self._timestamp < CACHE_TTL:
            return self._data
        return None

    def set(self, data: list[ApprovedTemplate]) -> None:
        self._data = data
        self._timestamp = time.time()

    def clear(self) -> None:
        self._data = None
        self._timestamp = 0


_cache = ApprovedTemplatesCache()


def load_repo_urls() -> list[str]:
    """Load whitelist URLs from config/approved_model_repos.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "approved_model_repos.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return []
    with open(config_path) as f:
        return yaml.safe_load(f) or []


def fetch_approved_templates_from_url(whitelist_url: str) -> list[ApprovedTemplate]:
    """Fetch and parse a single remote whitelist."""
    response = requests.get(whitelist_url, timeout=30)
    response.raise_for_status()
    data = yaml.safe_load(response.text)
    if data is None:
        return []
    return [ApprovedTemplate(**item) for item in data]


def fetch_all_approved_templates() -> list[ApprovedTemplate]:
    """Fetch from all configured URLs and merge results."""
    urls = load_repo_urls()
    all_templates: list[ApprovedTemplate] = []
    for url in urls:
        try:
            templates = fetch_approved_templates_from_url(url)
            all_templates.extend(templates)
        except Exception as e:
            logger.error(f"Failed to fetch whitelist from {url}: {e}")
    return all_templates


def get_approved_templates() -> list[ApprovedTemplate]:
    """Get approved templates with caching."""
    cached = _cache.get()
    if cached is not None:
        return cached

    templates = fetch_all_approved_templates()
    _cache.set(templates)
    return templates


def is_approved(url: str, version: str, approved: list[ApprovedTemplate]) -> bool:
    """Check if url+version is in the approved list."""
    for template in approved:
        if template.url == url and version in template.versions:
            return True
    return False


def get_git_ref(url: str, version: str, approved: list[ApprovedTemplate]) -> str | None:
    """Get the git ref for a given url and version name."""
    for template in approved:
        if template.url == url and version in template.versions:
            return template.versions[version]
    return None


def clear_cache() -> None:
    """Clear the approved templates cache."""
    _cache.clear()
