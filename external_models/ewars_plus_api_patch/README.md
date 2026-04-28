# ewars_plus_api patch

Local Dockerfile that overlays a one-file fix on top of
`maquins/ewars_plus_api:Upload`. Tracked in chap-core so the patched image
builds reproducibly from `docker compose up -d ewars_plus` without needing a
separate registry.

## What it fixes

In `Lag_Model_selection_ewars_By_District_api.R`, the per-fold CV predictions
are stacked with `foreach(aa, .combine = rbind) %do% get_preds(aa)`. The
upstream lag-selection step picks a different optimal lag per year, so each
per-year frame carries differently-named lag-suffixed covariate columns
(`mean_temperature_LAG12` vs `_LAG10` vs `_LAG7`). Base R `rbind` refuses to
stack frames with mismatched column names, so the call aborts with:

```
<simpleError in get_preds(aa): names do not match previous names>
```

and the wrapper surfaces it as `RuntimeError: EWARS API error from /Ewars_run:
500 - Internal server error`.

The patch replaces the two assembly calls with `dplyr::bind_rows(lapply(...))`,
which fills missing columns with `NA`. Downstream code does not read the
lag-suffixed columns, so the extra NAs are harmless.

See [CLIM-617](https://dhis2.atlassian.net/browse/CLIM-617) for the full
investigation and verification.

## How it's used

`compose.override.yml(.example)` points the `ewars_plus` service at this
directory:

```yaml
ewars_plus:
  build: ./external_models/ewars_plus_api_patch
  image: chap-core/ewars_plus_api:clim-617
  container_name: ewars_plus
  ports:
    - "3288:3288"
```

`docker compose up -d ewars_plus` will build the image on first run.

## Updating the upstream base

If a new `maquins/ewars_plus_api` tag is published, update the `FROM` line in
`Dockerfile` and re-test against the v3 Malawi reproduction payload (see
CLIM-615 / CLIM-617 for details).
