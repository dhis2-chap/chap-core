import logging
import re
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

COMMIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{7,40}")


@dataclass
class GithubUrl:
    owner: str
    repo_name: str
    commit: str  # can be commit or branch


def parse_github_url(github_url) -> GithubUrl:
    # trim trailing slash
    github_url = github_url.removesuffix("/")
    splitted_url = github_url.split("/")
    owner = splitted_url[3]
    repo_name = splitted_url[4]
    commit = "main"
    if "@" in repo_name:
        repo_name, commit = repo_name.split("@")

    return GithubUrl(owner=owner, repo_name=repo_name, commit=commit)


def resolve_commit_sha(github_url: str) -> str | None:
    """Resolve the ref in a github url to the commit sha it points to.

    A branch ref such as ``@main`` can move, but a sha cannot. A url that already
    has a commit sha (full or abbreviated) needs no lookup. Returns None if the lookup fails.
    """
    parsed = parse_github_url(github_url)
    if COMMIT_SHA_PATTERN.fullmatch(parsed.commit):
        return parsed.commit.lower()
    api_url = f"https://api.github.com/repos/{parsed.owner}/{parsed.repo_name}/commits/{parsed.commit}"
    try:
        fetched = requests.get(api_url, timeout=10)
        fetched.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not resolve commit sha for {github_url}: {e}")
        return None
    sha = fetched.json().get("sha")
    return str(sha) if sha is not None else None


def fetch_mlproject_content(github_url: str) -> str:
    parsed = parse_github_url(github_url)
    logger.warning(parsed)
    # Takes a github url, parses the MLProject file, returns an object with the correct information
    raw_mlproject_url = f"https://raw.githubusercontent.com/{parsed.owner}/{parsed.repo_name}/{parsed.commit}/MLproject"
    # fetch this MLProject file and parse it
    try:
        fetched = requests.get(raw_mlproject_url)
        assert fetched.status_code == 200, (
            f"Error fetching MLProject file from {raw_mlproject_url}: {fetched.status_code, fetched.content}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching MLProject file: {e}")
        return ""
    # TODO
    yaml_string = fetched.content
    return yaml_string.decode("utf-8")
