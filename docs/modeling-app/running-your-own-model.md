# Running Your Own Model

This guide covers how to run a model of your own alongside Chap in a Docker Compose deployment — for example a model developed locally for your own country or programme.

The supported way to do this is to package the model as a **chapkit service** and add it to your Compose stack as an extra service. Chap does not mount folders of model code into the running containers; instead your model runs as its own container with an HTTP interface, and Chap talks to it over the Compose network. This keeps the model's dependencies (R, Python, INLA, and so on) isolated from Chap's own image, and means you do not have to rebuild Chap when the model changes.

!!! note "Two different ways to add a model"
    This page is about models that run as a **service** next to Chap. If your model is an MLproject-style repository on GitHub, you do not need any of this — see [Managing models](managing-model-templates.md) instead.

## Prerequisites

- A working Chap installation, see [First-time Setup](fresh-installation.md)
- Your model packaged as a chapkit service. See the [chapkit documentation](https://dhis2-chap.github.io/chapkit/) for how to wrap an existing model, and [Chapkit](../external_models/chapkit.md) for the data format Chap sends.

## How model services attach to Chap

**Chapkit services register themselves with Chap on startup.** You give the service the address of Chap's registration endpoint, it announces itself, and Chap pulls in its model template and configurations automatically. Nothing needs to be listed in a configuration file, and Chap does not need rebuilding when you add or change a model.

This is how the bundled model services work, and it is the way to attach a model service. The rest of this page assumes it.

## Adding a self-registering model service

### 1. Create a Compose override file

Declare your service in a `compose.override.yml` file. Start from the shipped example:

```console
cp compose.override.yml.example compose.override.yml
```

Remove the sample services you do not need, and add your own.

!!! warning "Pass `-f compose.override.yml` explicitly"
    Compose only discovers `compose.override.yml` on its own when you run a bare `docker compose` with no `-f` flags at all. As soon as you pass any `-f` — as the installation guide does — the override file is ignored, silently and with no error, so your model never starts. Every command on this page therefore lists it explicitly, last, so its settings win.

### 2. Declare your service

```yaml
# compose.override.yml
services:
  my-model:
    image: ghcr.io/my-org/my-model:v1.0.0
    restart: unless-stopped
    ports:
      - "5003:8000"
    environment:
      SERVICEKIT_ORCHESTRATOR_URL: http://chap:8000/v2/services/$$register
      # Uncomment if chap has SERVICEKIT_REGISTRATION_KEY set:
      # SERVICEKIT_REGISTRATION_KEY: ${SERVICEKIT_REGISTRATION_KEY:-}
    depends_on:
      chap:
        condition: service_healthy
```

The important parts:

- **`SERVICEKIT_ORCHESTRATOR_URL`** points at Chap's registration endpoint using the Compose service name `chap`, not `localhost`. The `$$` is not a typo — Compose expands `$` as a variable, so `$$register` is how you write a literal `$register`.
- **`depends_on: chap: condition: service_healthy`** makes your model wait until Chap answers its health check, so the first registration attempt succeeds.
- **`ports`** is optional and only needed if you want to reach the model directly from the host for debugging. Chap itself reaches it over the internal network. Pick a host port that is not already taken — the bundled stack uses 8000 for `chap` and 5002 for `ewars`, and the sample services in `compose.override.yml.example` add 5001 (`chtorch`) and 3288 (`ewars_plus`).
- If your Chap deployment sets `SERVICEKIT_REGISTRATION_KEY` in `.env`, uncomment that line, or registration will be rejected.

`compose.ewars.yml` in the repository root is a working example of exactly this shape.

### 3. Start the stack

```console
docker compose -f compose.yml -f compose.chapkit.yml -f compose.override.yml up -d
```

Your service starts along with everything else. Use whichever base and overlays you normally use, with `compose.override.yml` last — see the [overlay reference](../webapi/docker-compose-doc.md#compose-file-reference). Pass the same `-f` flags to every later `down`, `build` and `logs` command in this stack.

### 4. Verify

Check that the service registered:

```console
curl http://localhost:8000/v2/services
```

Then check that it became a usable model:

```console
curl http://localhost:8000/v1/crud/configured-models
```

Your model should appear in both, and in the model list in the modeling app. The name comes from the service's own id as advertised by the chapkit image, not from the Compose service name.

## Building from a local model folder

If your model is not published as an image yet, point Compose at a local build context instead of an image. This is the closest equivalent to running a model straight from a folder:

```yaml
# compose.override.yml
services:
  my-model:
    build:
      context: ../my-model     # path to your model repository
    restart: unless-stopped
    environment:
      SERVICEKIT_ORCHESTRATOR_URL: http://chap:8000/v2/services/$$register
    depends_on:
      chap:
        condition: service_healthy
```

Rebuild after changing the model with:

```console
docker compose -f compose.yml -f compose.chapkit.yml -f compose.override.yml build my-model
docker compose -f compose.yml -f compose.chapkit.yml -f compose.override.yml up -d my-model
```

Note that the build context must be reachable from the Chap repository directory, and that a relative path like `../my-model` ties the deployment to your directory layout. For a server deployment, publishing an image to a registry is more robust.

## Lifecycle and troubleshooting

**Registered services must keep pinging.** A registration is valid for 30 seconds, and the chapkit service refreshes it automatically. If your model container stops, its registration expires and Chap eventually marks the model as archived, at which point it disappears from the model list. Archiving is not deletion: existing evaluations still resolve, and the model reappears when the service comes back and re-registers.

**A stopped model can linger in the list.** Expiry is only noticed the next time Chap syncs, which happens when some service registers or when the model template list is fetched. A model whose container died may stay visible until then.

**The model registered but does not appear as a model.** Chap syncs the service into its database as a side effect of registration, and that step is best-effort — if it fails, registration still succeeds. Fetch the model templates to force a fresh sync:

```console
curl http://localhost:8000/v1/crud/model-templates
```

**Registration is rejected.** Check whether `SERVICEKIT_REGISTRATION_KEY` is set in Chap's `.env` but missing from your service, or whether `CHAP_API_TOKEN` is set, which protects all endpoints. See [Service Registration](../webapi/service-registration.md) and [API Authentication](../webapi/api-authentication.md).

**Check the logs of both sides:**

```console
docker compose logs chap
docker compose logs my-model
```

## See also

- [Enabling Optional Model Services](enabling-optional-model-services.md) — the pre-built optional services shipped with Chap
- [Compose file reference](../webapi/docker-compose-doc.md#compose-file-reference) — which compose files stack and which are alternatives
- [Service Registration](../webapi/service-registration.md) — the registration and ping API
- [Chapkit](../external_models/chapkit.md) — the data format Chap sends to chapkit models
