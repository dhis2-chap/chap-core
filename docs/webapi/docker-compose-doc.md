# Setting up Chap REST-API locally

This is a short example for how to setup Chap-core locally as a service using docker-compose.

**Requirements:**

- Docker is installed **and running** on your computer (Installation instructions can be found at [https://docs.docker.com/get-started/get-docker/](https://docs.docker.com/get-started/get-docker/)).

## Step-by-Step Instructions:

1. Clone the Chap core repo by running `git clone https://github.com/dhis2-chap/chap-core.git`

2. Run the docker compose file with `docker compose -f compose.yml up`. The first time you do this, it can take a few minutes to finish. Once it's completed, it should have created the following docker services:

   - `redis` for receiving and queueing job requests
   - `worker` for executing the incoming work requests from queue
   - `chap` containing the main functionality and the rest-api
   - `postgres` for storing chap-related data

3. Check that the chap rest api works by going to http://localhost:8000/docs

## Compose file reference

The repository ships several compose files. `compose.yml` and `compose.ghcr.yml` are **base** files and are alternatives to each other — never stack them, because Compose appends list fields on overlay and you get duplicate `security_opt` / `cap_drop` entries that fail validation. Everything else is an overlay layered on top of a base with additional `-f` flags.

| File | Kind | Purpose |
|------|------|---------|
| `compose.yml` | base | Builds `chap` and `worker` from local source. The default for development and for the documented server install. |
| `compose.ghcr.yml` | base | Same services pulled as pre-built images from GHCR. Use *instead of* `compose.yml`. |
| `compose.chapkit.yml` | overlay | Umbrella overlay pulling in every bundled chapkit model service via the `include:` directive. Requires Compose v2.20+. |
| `compose.ewars.yml` | overlay | The EWARS chapkit model service on its own. Already included by `compose.chapkit.yml`. |
| `compose.override.yml.example` | overlay template | Optional extra services (`chtorch`, `ewars_plus`). Copy to `compose.override.yml`, which Compose merges automatically with no `-f` flag. |
| `compose.dev.yml` | overlay | Bind-mounts local source into `chap`, builds the worker from `Dockerfile.inla`, and exposes the postgres port on the host. |
| `compose.test.yml` | overlay | One-shot pytest container. |
| `compose.integration.test.yml` | overlay | Frontend emulator running the end-to-end database flow. |
| `compose.r-model.integration.test.yml` | overlay | End-to-end flow for an R-based model. |

Common combinations:

```console
# Base only
docker compose up -d

# With all bundled model services (what the installation guide uses)
docker compose -f compose.yml -f compose.chapkit.yml up -d

# Development, with local source bind-mounted
docker compose -f compose.yml -f compose.dev.yml up -d

# Pre-built images instead of a local build
docker compose -f compose.ghcr.yml up -d
```

Docker Compose does not remember which overlays you used, so pass the same `-f` flags to every subsequent `down`, `build` and `logs` command in that stack. The `make restart`, `make force-restart` and `make chap-version` targets already carry the `compose.yml` + `compose.chapkit.yml` pair.

To add a model service of your own, see [Running Your Own Model](../modeling-app/running-your-own-model.md).
