# CHAP Helm Chart

Umbrella chart for deploying CHAP (Climate Health Analysis Platform). It bundles the following sub-charts:

- **chap-api** — REST API
- **chap-worker** — Celery worker
- **chap-db** — PostgreSQL via CloudNativePG
- **valkey** — Valkey (Redis-compatible) message broker

## Deploy

### Skaffold (recommended for local development)

The easiest way to deploy CHAP locally is to use the skaffold.yaml file found at the root of the repository:

```shell
skaffold run
```

### Helm

```shell
helm dependency update charts/chap
helm upgrade --install chap charts/chap \
    --namespace chap \
    --create-namespace
```

## Dependencies

### Valkey

Valkey is deployed as a sub-chart dependency by default. To use an external Valkey instance, disable the
sub-chart and configure the connection:

```yaml
valkey:
  enabled: false

chap-api:
  valkey:
    host: <your-valkey-host>
    port: 6379
    existingSecret: <secret-name>
    secretKeys:
      password: <key-in-secret>

chap-worker:
  valkey:
    host: <your-valkey-host>
    port: 6379
    existingSecret: <secret-name>
    secretKeys:
      password: <key-in-secret>
```

### PostgreSQL

PostgreSQL is deployed by default using the CloudNativePG operator, which must be installed on the cluster:

```shell
helm repo add cnpg https://cloudnative-pg.github.io/charts
helm repo update
helm upgrade --install cnpg \
    --namespace cnpg-system \
    --create-namespace \
    cnpg/cloudnative-pg
```

To use an external PostgreSQL server instead, disable the CloudNativePG cluster, provide the external
credentials in `chap-db` (which will create the shared secret), and override the host for the API and
worker (since the default points to the CNPG service):

```yaml
chap-db:
  postgresql:
    cnpg:
      cluster:
        enabled: false
    external:
      enabled: true
      host: <your-postgres-host>
      user: <username>
      password: <password>

chap-api:
  postgres:
    host: <your-postgres-host>

chap-worker:
  postgres:
    host: <your-postgres-host>
```

See [values.yaml](./values.yaml) for all available configuration options.

## Connect from DHIS2

To connect DHIS2 to CHAP:

1. Deploy DHIS2 with the following in `dhis.conf`: `route.remote_servers_allowed=http://*`
2. Run analytics in DHIS2
3. Install the Modeling app from App Hub
4. Configure the connection in the Modeling app. With the default Skaffold setup the URL is `http://chap.chap.svc:8000/**`
