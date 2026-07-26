# CHAP Helm Chart

Deploys CHAP (Climate Health Analysis Platform) as a single chart with the following components:

- **api** — REST API
- **worker** — Celery worker
- **db** — PostgreSQL via CloudNativePG
- **valkey** — Valkey (Redis-compatible) message broker, deployed as a subchart

All internal wiring (hostnames, secrets) between the components is derived from the release name, so no cross-component configuration is needed. Each component can be disabled and replaced with an external service.

## Deploy

### Skaffold (recommended for local development)

The easiest way to deploy CHAP locally is to use the skaffold.yaml file found at the root of the repository. It requires the `CHAP_DB_PASSWORD` and `REDIS_PASSWORD` environment variables to be set:

```shell
skaffold run
```

### Helm

PostgreSQL is provisioned through the CloudNativePG operator, which must be installed on the cluster first:

```shell
helm repo add cnpg https://cloudnative-pg.github.io/charts
helm repo update
helm upgrade --install cnpg \
    --namespace cnpg-system \
    --create-namespace \
    cnpg/cloudnative-pg
```

The chart has no default credentials, so the database and Valkey passwords must be provided:

```shell
helm dependency update charts/chap
helm upgrade --install chap charts/chap \
    --namespace chap \
    --create-namespace \
    --set db.password=<password> \
    --set valkey.auth.aclUsers.default.password=<password>
```

## External services

### PostgreSQL

To use an external PostgreSQL server instead of the CloudNativePG cluster:

```yaml
db:
  enabled: false

externalDatabase:
  host: <your-postgres-host>
  username: <username>
  password: <password>       # or use existingSecret
```

### Valkey

To use an external Valkey/Redis instance instead of the bundled subchart:

```yaml
valkey:
  enabled: false

externalValkey:
  host: <your-valkey-host>
  password: <password>       # or use existingSecret
```

## Instance Manager

When deploying via IM, set the IM-required labels once under `global.commonLabels` (and `valkey.commonLabels`, since the valkey subchart does not read global values). They propagate to all resources, pods and PVCs of every component. Components are distinguished by the `app.kubernetes.io/component` label (`api`, `worker`, `db`), and per-component pod labels (e.g. `im-type`) can be added via `api.podLabels`, `worker.podLabels`, `db.podLabels` and `valkey.podLabels`:

```yaml
global:
  commonLabels: &common
    im: "true"
    # ... other im-* labels

valkey:
  commonLabels: *common
  podLabels:
    im-type: chap-valkey

api:
  podLabels:
    im-type: chap-api

worker:
  podLabels:
    im-type: chap-worker

db:
  podLabels:
    im-type: chap-db
```

See [values.yaml](./values.yaml) for all available configuration options.

## Connect from DHIS2

Setting `dhis2.enabled=true` (with `dhis2.hostname` and credentials) runs a job after each install/upgrade that registers CHAP as a route in the DHIS2 instance. To set up manually:

1. Deploy DHIS2 with the following in `dhis.conf`: `route.remote_servers_allowed=http://*`
2. Run analytics in DHIS2
3. Install the Modeling app from App Hub
4. Configure the connection in the Modeling app. With the default Skaffold setup the URL is `http://chap-api.chap.svc:8000/**`
