# Welcome to the Chap modelling platform!

[![CI](https://github.com/dhis2-chap/chap-core/actions/workflows/ci-test-python-install.yml/badge.svg)](https://github.com/dhis2-chap/chap-core/actions/workflows/ci-test-python-install.yml)
[![PyPI version](https://img.shields.io/pypi/v/chap-core)](https://pypi.org/project/chap-core/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://chap.dhis2.org/chap-modeling-platform/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/dhis2-chap/chap-core/badge)](https://scorecard.dev/viewer/?uri=github.com/dhis2-chap/chap-core)

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
echo "SERVICEKIT_REGISTRATION_KEY=$(openssl rand -hex 32)" >> .env
docker compose up
```

### Rebuilding after a source change

`docker compose up` will reuse an existing `chap-core-chap` image if one is
already built — it does not automatically rebuild when you edit source. If
you see a stale `chap_core.__version__` or a fix that clearly didn't land
inside the running container, use one of:

```shell
make docker-up       # build what changed and start in the foreground; Ctrl+C stops it
make docker-down     # stop and remove the containers (keeps the database)
make docker-restart  # docker-down, then docker-up
make docker-reset    # delete the database, rebuild without cache, start (WIPES VOLUMES)
make docker-logs     # follow the logs from a second terminal, or of a stack started with up -d
make docker-version  # print the chap_core version running inside the container
```

`make docker-up` is the right one most of the time: it rebuilds only what changed, keeps the
database, and shows the logs until you stop it with Ctrl+C. `make docker-reset` also deletes the
Postgres volume, so reach for it only when you need a clean slate. All of them use `compose.yml`
with `compose.chapkit.yml`, so the chapkit models start too.

### Running marketplace models

Chapkit-based models are installed from the
[Chap Marketplace](https://github.com/dhis2-chap/chap-marketplace)
with `chap-admin`, once chap is running. A fresh deployment needs this manual
step after the first `docker compose up -d`:

```shell
chap-admin install-all
```

It starts every model service with a verified stable version, registers it in
chap from what the running service reports, and adds the model's verified
configurations. `chap-admin install chapkit_simple_multistep_model` does the same
for one model. Include the overlay it writes in later Compose commands:
`docker compose -f compose.yml -f compose.marketplace.yml up -d`. See the
[CLI setup guide](docs/chap-cli/chap-core-cli-setup.md) for the details.

Do not start models with `compose.chapkit.yml` or `compose.ewars.yml` in a
deployment. A service started that way registers a model template but gets no
configurations, so it cannot be run. Those overlays are kept for development.

## Deploy on Kubernetes

The Helm chart for Kubernetes lives in [chap-kubernetes](https://github.com/dhis2-chap/chap-kubernetes).
