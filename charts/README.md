# CHAP Helm Charts

This directory contains Helm charts for deploying CHAP (Climate Health Analysis Platform) on Kubernetes.

## Structure

CHAP uses a master/worker architecture. To support deployment on IM (Instance Manager), where individual
components need to be referenced separately, the deployment is split into multiple charts gathered under
a single umbrella chart.

| Chart | Description |
|---|---|
| [chap](./chap) | Umbrella chart — use this for standalone deployments |
| [chap-api](./chap-api) | REST API deployment |
| [chap-worker](./chap-worker) | Celery worker deployment |
| [chap-db](./chap-db) | PostgreSQL database via CloudNativePG |

## Standalone deployment

For deploying outside of IM, use the umbrella chart (`charts/chap`). It bundles all sub-charts and
their dependencies (including Valkey) for a single-command deployment.

See [charts/chap/README.md](./chap/README.md) for details.

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
