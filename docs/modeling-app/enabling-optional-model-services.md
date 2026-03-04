# Enabling Optional Model Services

Some models require an external service to be running alongside CHAP. These services are not started by default but can be enabled using a Docker Compose overlay file.

## Available Optional Services

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `ewars_plus` | `maquins/ewars_plus_api:Upload` | 3288 | EWARS Plus R-based prediction API |
| `chtorch` | `ghcr.io/dhis2-chap/chtorch:chapkit2-8f17ee3` | 5001 | Deep learning model using chapkit |

## Setup

CHAP ships with a `compose.override.yml.example` file that defines these optional services. Docker Compose automatically merges `compose.override.yml` with `compose.yml` when both are present.

### 1. Copy the overlay file

```console
cp compose.override.yml.example compose.override.yml
```

### 2. Edit the overlay (optional)

Open `compose.override.yml` and remove any services you do not need. For example, to enable only EWARS Plus:

```yaml
services:
  ewars_plus:
    image: maquins/ewars_plus_api:Upload
    container_name: ewars_plus
    ports:
      - "3288:3288"
```

### 3. Add the model to configured models

The model also needs to be registered so that CHAP seeds it on startup. Create or edit a YAML file in `config/configured_models/` (do **not** edit `default.yaml`):

```yaml
# config/configured_models/local.yaml
- url: https://github.com/dhis2-chap/ewars_plus_python_wrapper/
  versions:
    v1: "@modeling_app_test"
```

See [Managing models](managing-model-templates.md) for details on the configured models format.

### 4. Rebuild and start

After adding the overlay and the model configuration, rebuild the CHAP images (so the new config is included) and start all services:

```console
docker compose build chap worker
docker compose up -d
```

### 5. Verify

Check that the service is running and the model appears in the API:

```console
docker compose ps
curl http://localhost:8000/v1/crud/models
```

The model (e.g. `ewars_plus`) should appear in the list of configured models.
