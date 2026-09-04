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
| `compose.ghcr.yml` | base | Same services pulled as pre-built images from GHCR. Use *instead of* `compose.yml`. Self-contained: download this one file and run it without a checkout or an `.env`. |
| `compose.chapkit.yml` | overlay | Umbrella overlay pulling in every bundled chapkit model service via the `include:` directive. Requires Compose v2.20+. |
| `compose.ewars.yml` | overlay | The EWARS chapkit model service on its own. Already included by `compose.chapkit.yml`. |
| `compose.override.yml.example` | overlay template | Optional extra services (`chtorch`, `ewars_plus`). Copy to `compose.override.yml`. Compose merges that file automatically **only** when no `-f` flag is used; with any `-f` flag you must list it explicitly, last. |
| `compose.dev.yml` | overlay | Bind-mounts local source into `chap`, builds the worker from `Dockerfile.inla`, and exposes the postgres port on the host. |
| `compose.test.yml` | overlay | One-shot pytest container. |
| `compose.integration.test.yml` | overlay | Frontend emulator running the end-to-end database flow. |
| `compose.r-model.integration.test.yml` | overlay | End-to-end flow for an R-based model. |

Common combinations:

```console
# Base only (plus compose.override.yml, if one exists)
docker compose up -d

# With all bundled model services (what the installation guide uses)
docker compose -f compose.yml -f compose.chapkit.yml up -d

# Development, with local source bind-mounted
docker compose -f compose.yml -f compose.dev.yml up -d

# Pre-built images instead of a local build
docker compose -f compose.ghcr.yml up -d
```

### Deploying a release without a checkout

`compose.ghcr.yml` is the only file you need on a server. It pulls pre-built
images and every setting has a working default, so it runs exactly as
downloaded -- no checkout, no `.env`, no edits:

```console
curl -O https://raw.githubusercontent.com/dhis2-chap/chap-core/master/compose.ghcr.yml
docker compose -f compose.ghcr.yml up -d
```

#### Available settings

All optional. Put them in an `.env` file beside the compose file rather than
inline on the command line: Compose reads `.env` on every command, so a later
`pull` or `up` keeps the same values, whereas an inline `VAR=x docker compose
...` applies to that one command and silently reverts afterwards.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHAP_IMAGE_TAG` | `latest` | Tag for both the `chap-core` and `chap-worker` images. Set a release tag (for example `v1.2.3`) to pin a version. The tag must exist in [GHCR](https://github.com/orgs/dhis2-chap/packages); release tags are published by the image build workflow. |
| `POSTGRES_USER` | `chap` | Database user. |
| `POSTGRES_PASSWORD` | `chap` | Database password. Postgres is never published to the host, but override this for anything beyond a local trial. It is interpolated into a database URI, so it must be URL-safe -- no `@`, `:`, `/`, `?`, `#` or `%`. |
| `POSTGRES_DB` | `chap_core` | Database name. |
| `CHAP_DATABASE_URL` | composed from the three above | Full, percent-encoded database URL. Takes precedence, and is how you use a password containing reserved characters. |
| `CHAP_ROOT_PATH` | empty | Path prefix when serving behind a reverse proxy. |
| `CHAP_API_TOKEN` | empty | API token. Unset means no authentication. |

Pinning a release and setting a password with reserved characters:

```console
cat > .env <<'EOF'
CHAP_IMAGE_TAG=v1.2.3
POSTGRES_PASSWORD=str@ng/pass
CHAP_DATABASE_URL=postgresql://chap:str%40ng%2Fpass@postgres:5432/chap_core
EOF
docker compose -f compose.ghcr.yml up -d
```

Pass the same `-f compose.ghcr.yml` flag to every later `down`, `pull` and
`logs` command in that stack.

Docker Compose does not remember which overlays you used, so pass the same `-f` flags to every subsequent `down`, `build` and `logs` command in that stack. The `make restart`, `make force-restart` and `make chap-version` targets already carry the `compose.yml` + `compose.chapkit.yml` pair.

To add a model service of your own, see [Running Your Own Model](../modeling-app/running-your-own-model.md).
