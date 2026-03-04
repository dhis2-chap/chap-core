# CHAP Helm Chart

## Deploy

The easiest way to deploy CHAP is to use the skaffold.yaml file found [here](/skaffold.yaml).

```shell
skaffold run
```

## Dependencies

The chart depends on Valkey and Postgresql.

Valkey can be deployed as a dependency of this chart and is done so by default. However, if you wish to deploy Valkey
separately, you can do so. Please see the [values.yaml](./values.yaml) file for more information.

Postgresql is by default deployed using the CloudNativePG operator (which needs to be installed on the cluster).
However, an external Postgresql server can also be used. Please see the [values.yaml](./values.yaml) file for more
information.

### Install CloudNativePG operator

```shell
helm upgrade --install cnpg \
    --namespace cnpg-system \
    --create-namespace \
    cnpg/cloudnative-pg
```

## Connect from DHIS2

In order to connect from DHIS2 there are several things which needs to be done

* Deploy DHIS2 with the following in the dhis.conf file: `route.remote_servers_allowed=http://*`
* Run analytics
* Install the Modeling app from App Hub
* Configure the connection in the Modeling app. If you install using the Skaffold file the default should be `http://chap.chap.svc:8000/**`

## Release

Bump the version in [Chart.yaml](./Chart.yaml), commit and push.
**NOTE: do not create a tag yourself!**

Our release workflow will then using [Helm chart releaser action](https://github.com/helm/chart-releaser-action)

* create a tag `CHAP-<version>`
* create a [release](https://github.com/dhis2-sre/dhis2-core-helm/releases) associated with the new tag
* commit an updated index.yaml with the new release
* redeploy the GitHub pages to serve the new index.yaml

Note: there might be a slight delay between the release and the `index.yaml` file being updated as GitHub pages have to
be re-deployed.
