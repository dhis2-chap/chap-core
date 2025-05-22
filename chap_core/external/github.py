from dataclasses import dataclass
import logging

import requests

logger = logging.getLogger(__name__)


@dataclass
class GithubUrl:
    owner: str
    repo_name: str
    commit: str  # can be commit or branch


def parse_github_url(github_url) -> GithubUrl:
    splitted_url = github_url.split("/")
    owner = splitted_url[3]
    repo_name = splitted_url[4]
    commit = "main"
    if "@" in repo_name:
        repo_name, commit = repo_name.split("@")

    return GithubUrl(owner=owner, repo_name=repo_name, commit=commit)


def fetch_mlproject_content(github_url: str) -> str:
    parsed = parse_github_url(github_url)
    logger.info(parsed)
    # Takes a github url, parses the MLProject file, returns an object with the correct information
    raw_mlproject_url = f"https://raw.githubusercontent.com/{parsed.owner}/{parsed.repo_name}/{parsed.commit}/MLproject"
    # fetch this MLProject file and parse it
    try:
        fetched = requests.get(raw_mlproject_url)
        assert (
            fetched.status_code == 200
        ), f"Error fetching MLProject file from {raw_mlproject_url}: {fetched.status_code, fetched.content}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching MLProject file: {e}")
        return None
    # TODO
    yaml_string = fetched.content
    return yaml_string
