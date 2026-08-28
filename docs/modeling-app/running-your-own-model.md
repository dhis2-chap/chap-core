# Running Your Own Model

This guide covers how to run a model of your own alongside Chap in a Docker Compose deployment — for example a model developed locally for your own country or programme.

The supported way to do this is to package the model as a **chapkit service** and add it to your Compose stack as an extra service. Chap does not mount folders of model code into the running containers; instead your model runs as its own container with an HTTP interface, and Chap talks to it over the Compose network. This keeps the model's dependencies (R, Python, INLA, and so on) isolated from Chap's own image, and means you do not have to rebuild Chap when the model changes.

!!! note "Two different ways to add a model"
    This page is about models that run as a **service** next to Chap. If your model is an MLproject-style repository on GitHub, you do not need any of this — see [Managing models](managing-model-templates.md) instead.

## Prerequisites

- A working Chap installation, see [First-time Setup](fresh-installation.md)
- Your model packaged as a chapkit service. See the [chapkit documentation](https://dhis2-chap.github.io/chapkit/) for how to wrap an existing model, and [Chapkit](../external_models/chapkit.md) for the data format Chap sends.

## How model services attach to Chap

There are two mechanisms, and it is worth knowing which one you are using:

| | Self-registration | Seeding from config |
|---|---|---|
| Who initiates | The model service calls Chap on startup | Chap reads a YAML file on startup |
| Configuration | `SERVICEKIT_ORCHESTRATOR_URL` on the model service | `config/configured_models/*.yaml` in the Chap repo |
| Requires rebuilding Chap | No | Yes, the config is baked into the image |
| Startup order | Model service can start at any time | Model service must be healthy before Chap seeds |
| Survives model restarts | Yes, it re-registers | Only across a Chap restart |

**Self-registration is the recommended path** and the one used by the bundled model services. Use the config-seeding path only for a service that cannot be given the registration environment variables.

## Adding a self-registering model service

### 1. Create a Compose override file

Chap picks up a `compose.override.yml` file automatically, with no extra `-f` flag. Start from the shipped example:

```console
cp compose.override.yml.example compose.override.yml
```

Remove the sample services you do not need, and add your own.

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
- **`ports`** is optional and only needed if you want to reach the model directly from the host for debugging. Chap itself reaches it over the internal network. Pick a host port that is not already taken — the bundled services use 5001 and 5002.
- If your Chap deployment sets `SERVICEKIT_REGISTRATION_KEY` in `.env`, uncomment that line, or registration will be rejected.

`compose.ewars.yml` in the repository root is a working example of exactly this shape.

### 3. Start the stack

```console
docker compose -f compose.yml -f compose.chapkit.yml up -d
```

`compose.override.yml` is merged in automatically, so your service starts along with everything else. Use whichever base and overlays you normally use — see the [overlay reference](../webapi/docker-compose-doc.md#compose-file-reference).

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
docker compose build my-model && docker compose up -d my-model
```

Note that the build context must be reachable from the Chap repository directory, and that a relative path like `../my-model` ties the deployment to your directory layout. For a server deployment, publishing an image to a registry is more robust.

## Alternative: seeding from configuration

If your service cannot self-register, Chap can be told about it at startup. Add a YAML file to `config/configured_models/` — do **not** edit `default.yaml`, which is overwritten on upgrade:

```yaml
# config/configured_models/local.yaml
- url: http://my-model:8000
  uses_chapkit: true
  versions:
    v1: "/v1"
  configurations:
    default:
      user_option_values: {}
```

`url` is the service's address on the Compose network, and `uses_chapkit: true` tells Chap to read the model template over HTTP rather than fetching an `MLProject.yaml` from GitHub. See [the configured models reference](https://github.com/dhis2-chap/chap-core/blob/master/config/configured_models/README.md) for the full file format.

Because this configuration is baked into the Chap image, you must rebuild after editing it:

```console
docker compose build chap worker
docker compose up -d
```

Chap waits up to 30 seconds for the service to report healthy while seeding. If it does not respond in time, the model is skipped with an error in the Chap log and startup continues — restart Chap once the service is up.

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
