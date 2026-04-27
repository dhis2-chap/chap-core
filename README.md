# Welcome to the Chap modelling platform!

[![CI](https://github.com/dhis2-chap/chap-core/actions/workflows/ci-test-python-install.yml/badge.svg)](https://github.com/dhis2-chap/chap-core/actions/workflows/ci-test-python-install.yml)
[![PyPI version](https://img.shields.io/pypi/v/chap-core)](https://pypi.org/project/chap-core/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://chap.dhis2.org/chap-modeling-platform/)

This is the main repository for the Chap modelling platform.

[Read more about the Chap project here](https://chap.dhis2.org/about/)

## Code documentation

The main documentation for the modelling platform is located at [https://chap.dhis2.org/chap-modeling-platform/](https://chap.dhis2.org/chap-modeling-platform/).

## Development / contribution

Information about how to contribute to the Chap Modelling Platform: [https://chap.dhis2.org/chap-modeling-platform/contributor/](https://chap.dhis2.org/chap-modeling-platform/contributor/).

## Issues/Bugs

If you find any bugs or issues when using this code base, we appreciate it if you file a bug report here: https://github.com/dhis2-chap/chap-core/issues/new

## Launch development instance using Docker

```shell
cp .env.example .env
docker compose up
```

### Rebuilding after a source change

`docker compose up` will reuse an existing `chap-core-chap` image if one is
already built — it does not automatically rebuild when you edit source. If
you see a stale `chap_core.__version__` or a fix that clearly didn't land
inside the running container, use one of:

```shell
make restart       # down && up -d --build (preserves volumes incl. chap-db)
make force-restart # down -v && build --no-cache && up (WIPES VOLUMES)
make chap-version  # print the chap_core version running inside the container
```

`make restart` is the right hammer 90% of the time. `make force-restart`
also wipes the Postgres volume, so reach for it only when you need a clean
slate. `make chap-version` is also printed automatically at the end of
`make restart` so version drift is visible at a glance.

### Running with the chapkit EWARS overlay

The chapkit-based EWARS model ships as an opt-in compose overlay at
`compose.ewars.yml`. Layer it onto `compose.yml` (not `compose.ghcr.yml`
— those two are alternatives, not stackable) to run chap-core with the
ewars service already self-registered:

```shell
docker compose -f compose.yml -f compose.ewars.yml up -d
```
