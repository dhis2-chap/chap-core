# CHAP Helm Charts

This directory contains Helm charts for deploying CHAP (Climate Health Analysis Platform) on Kubernetes.

## Structure

| Chart | Description |
|---|---|
| [chap](./chap) | The CHAP chart — deploys the API, worker, database and Valkey as components of a single release |
| [chap-api](./chap-api) | Legacy: REST API deployment |
| [chap-worker](./chap-worker) | Legacy: Celery worker deployment |
| [chap-db](./chap-db) | Legacy: PostgreSQL database via CloudNativePG |

The legacy component charts exist only because the current Instance Manager deploys each CHAP component as a separate release. Once the IM component model (which deploys the `chap` chart as a single stack) is rolled out, they will be removed. New deployments should use the `chap` chart.

See [charts/chap/README.md](./chap/README.md) for deployment instructions.

## Release

Each chart is released independently. To release a chart, bump its version in the corresponding
`Chart.yaml`, commit and push to master.
**NOTE: do not create a tag yourself.**

The release workflow uses [Helm chart releaser action](https://github.com/helm/chart-releaser-action) to:

- Create a tag `<chart-name>-<version>` (e.g., `chap-0.3.2`)
- Create a [release](https://github.com/dhis2-chap/chap-core/releases) associated with the new tag
- Commit an updated `index.yaml` with the new release
- Redeploy GitHub Pages to serve the updated `index.yaml`

Note: there may be a slight delay between the release and the `index.yaml` being updated as GitHub Pages need to be re-deployed.
