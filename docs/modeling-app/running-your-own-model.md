# Running Your Own Model

This guide covers how to run a model of your own alongside Chap in a Docker Compose deployment — for example a model developed locally for your own country or programme.

The supported way to do this is to package the model as a **chapkit service** and add it to your Compose stack as an extra service. Chap does not mount folders of model code into the running containers; instead your model runs as its own container with an HTTP interface, and Chap talks to it over the Compose network. This keeps the model's dependencies (R, Python, INLA, and so on) isolated from Chap's own image, and means you do not have to rebuild Chap when the model changes.

!!! note "Two different ways to add a model"
    This page is about models that run as a **service** next to Chap. If your model is an MLproject-style repository on GitHub, you do not need any of this — see [Managing models](managing-model-templates.md) instead.

## Prerequisites

- A working Chap installation, see [First-time Setup](fresh-installation.md)
- Your model packaged as a chapkit service. See the [chapkit documentation](https://dhis2-chap.github.io/chapkit/) for how to wrap an existing model, and [Chapkit](../external_models/chapkit.md) for the data format Chap sends.

## How model services attach to Chap

**Chapkit services register themselves with Chap on startup.** You give the service the address of Chap's registration endpoint and it announces itself, so Chap knows where it runs and whether it is alive. Registration alone does not make it a model, though: Chap only lists models that were registered on purpose, together with the configurations they should run with. For a marketplace model, `chap-admin install` does all of this for you, including the Compose service, so you do not need this page. For your own image, you declare the service yourself as shown below and then register it as a model from its running service. Chap does not need rebuilding either way.

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
- No `networks` key is needed. Model services land on the default Compose network, which reaches `chap` and `worker` but not the Celery broker or the database, so a model container cannot enqueue tasks or read data on its own.
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

### 5. Register it as a model

A registered service is not a model yet. Store its template from the running service, using the id it registered under (the service's own id as advertised by the chapkit image, not the Compose service name), and give it a configuration:

```console
curl -X POST http://localhost:8000/v1/crud/model-templates/from-service \
  -H 'Content-Type: application/json' -d '{"serviceId": "my-model"}'
curl -X POST http://localhost:8000/v1/crud/configured-models \
  -H 'Content-Type: application/json' \
  -d '{"name": "default", "modelTemplateId": <id from the previous response>, "userOptionValues": {}}'
```

Add `-H 'Authorization: Bearer $CHAP_API_TOKEN'` when the deployment requires a token. Both calls can be repeated: a stored template version is returned unchanged, and an identical configuration is not stored twice. The template is stored under the version the service reports, from the commit it reports; a rebuilt image under the same version is refused, so bump the service version when the model changes.

`chap-admin install my_model --image ghcr.io/my-org/my-model:v1.0.0 --accept-risk` makes the same calls after writing and starting the service itself, if you would rather not hand-write the Compose service.

Then check that it became a usable model:

```console
curl http://localhost:8000/v1/crud/configured-models
```

Your model should appear there and in the model list in the modeling app.

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

**Registered services must keep pinging.** A registration is valid for 30 seconds, and the chapkit service refreshes it automatically. If your model container stops, its registration expires and the template's `healthStatus` in `GET /v1/crud/model-templates` goes from `live` to empty. The model stays listed; runs against it fail until the service is back. To take a model out of the pickers, retire it with `DELETE /v1/crud/model-templates/{id}` (what `chap-admin uninstall` does). Retiring is not deletion: existing evaluations still resolve, and storing the same version again shows the model again.

**The model registered but does not appear as a model.** Registration does not create the model; step 5 above does. If the template is stored but `healthStatus` is `revision_mismatch`, the running image was built from another commit than the one stored under its version: bump the service version and register it again.

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
