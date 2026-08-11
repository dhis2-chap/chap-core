# CHAP Helm Charts

This directory contains Helm charts for deploying CHAP (Climate Health Analysis Platform) on Kubernetes.

## Structure

The [chap](./chap) chart deploys the API, worker, database and Valkey as components of a single release. See [charts/chap/README.md](./chap/README.md) for deployment instructions.

The previous per-component charts (`chap-api`, `chap-worker`, `chap-db`) have been removed; their released versions remain available through the chart repository index for existing deployments.

## Release

To release the chart, bump the version in [chap/Chart.yaml](./chap/Chart.yaml), commit and push to master.
**NOTE: do not create a tag yourself.**

The release workflow uses [Helm chart releaser action](https://github.com/helm/chart-releaser-action) to:

- Create a tag `<chart-name>-<version>` (e.g., `chap-0.3.2`)
- Create a [release](https://github.com/dhis2-chap/chap-core/releases) associated with the new tag
- Commit an updated `index.yaml` with the new release
- Redeploy GitHub Pages to serve the updated `index.yaml`

Note: there may be a slight delay between the release and the `index.yaml` being updated as GitHub Pages need to be re-deployed.
