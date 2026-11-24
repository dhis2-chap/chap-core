"""Sync chap_core/terminology/terminology.yaml with terminology.yaml in dhis2-chap/user-stories.

Run from the repo root:

    uv run python -m scripts.sync_terminology           # rewrite the copy if the source has changed
    uv run python -m scripts.sync_terminology --check   # change nothing, exit 1 if the copy differs

A source file that fails validation never replaces the copy. Exit codes: 0 when the copy is
up to date or was updated, 1 when ``--check`` finds a difference, 2 when the fetch or the
validation fails. ``GITHUB_TOKEN``, when set, is sent to the GitHub API.

Needs only the standard library and PyYAML.
"""

import argparse
import json
import os
import sys
import urllib.request

import yaml

from chap_core.terminology import COPY_PATH, SOURCE_PATH, SOURCE_REPOSITORY, split_copy, validate_terms

COMMITS_URL = f"https://api.github.com/repos/{SOURCE_REPOSITORY}/commits?path={SOURCE_PATH}&sha=main&per_page=1"
RAW_URL = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{{commit}}/{SOURCE_PATH}"
HEADER = f"""\
# Copy of {SOURCE_PATH} from https://github.com/{SOURCE_REPOSITORY}, synced by
# scripts/sync_terminology.py. Do not edit it here: change a term with a pull
# request to {SOURCE_REPOSITORY}.
# source-commit: {{commit}}
# source-date: {{date}}

"""


def fetch(url: str, token: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return bytes(response.read())


def fetch_source() -> tuple[str, str, str]:
    """The commit that last changed the source file, that commit's date, and the file at that commit."""
    latest = json.loads(fetch(COMMITS_URL, token=os.environ.get("GITHUB_TOKEN")))[0]
    commit = latest["sha"]
    content = fetch(RAW_URL.format(commit=commit)).decode("utf-8")
    return commit, latest["commit"]["committer"]["date"], content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if the copy differs")
    args = parser.parse_args(argv)

    try:
        commit, date, content = fetch_source()
    except (OSError, ValueError, LookupError) as error:
        print(f"error: could not fetch the terminology: {error!r}", file=sys.stderr)
        return 2
    if COPY_PATH.exists() and split_copy(COPY_PATH.read_bytes().decode("utf-8"))[1] == content:
        print(f"{COPY_PATH} is up to date with {SOURCE_REPOSITORY}")
        return 0
    if args.check:
        print(f"{COPY_PATH} differs from {SOURCE_PATH} in {SOURCE_REPOSITORY} at {commit}")
        return 1

    try:
        errors = validate_terms(yaml.safe_load(content))
    except yaml.YAMLError as error:
        errors = [f"not valid YAML: {error}"]
    if errors:
        for error in errors:
            print(f"error: {SOURCE_PATH} at {commit}: {error}", file=sys.stderr)
        print(f"{COPY_PATH} was left unchanged", file=sys.stderr)
        return 2
    COPY_PATH.write_bytes((HEADER.format(commit=commit, date=date) + content).encode("utf-8"))
    print(f"Updated {COPY_PATH} to {SOURCE_REPOSITORY} at {commit} ({date})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
