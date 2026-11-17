# Translating API text

Chap translates metric and model descriptions, plot metadata and chart labels.
Metric and model names stay as written, so they match the literature and the
model author's naming. API error and status messages stay English. Clients
send the user's UI language in `Accept-Language`, for example `fr` or `fr-CA`. Chap honors quality weights and falls back to a base language,
then English. Missing and fuzzy translations also fall back to the English source
string. Responses include `Content-Language` and `Vary: Accept-Language`.

Translation happens in the request context, including synchronous handlers.
Model metadata is translated during response serialization; stored values, model
versions, IDs, configuration names and scores stay unchanged. The catalog includes
built-in model descriptions. Text supplied by external models is translated only when
the catalog contains its exact source string. User data, error messages and
worker logs are not translated. CLI commands
and workers default to English. French translations and HISP WCA review are tracked
in [CLIM-1895](https://dhis2.atlassian.net/browse/CLIM-1895).

## Adding strings

Use `N_("English text")` from `chap_core.i18n` for shared metadata, such as a
metric specification or plot registration. It marks the text for Babel without
translating it at import time. Use `gettext("English text")` when building a
response or plot during a request. Do not pass an f-string to `gettext`: Babel
needs the literal source to extract it.

`make lint` regenerates `chap_core/locales/en.pot`; commit it with the code change.
`make check` rejects stale catalogs in CI. To extract without running other checks:

<!-- Cannot test: rewrites the committed source catalog. -->
```console
make extract-translations
```

Translation files live at `chap_core/locales/<language>.po`. They ship inside the
Python package and are compiled in memory once per process, so deployments need no
separate compilation step. Restart Chap after updating catalogs.

## Transifex

Chap-core is the `chap-core-en-pot` resource in the `hisp-uio/modeling-app`
project, next to the chap-frontend resources, configured in `.tx/config`. For
chap-frontend, dhis2-bot opens PRs with translated PO files from Transifex; ask the
DHIS2 apps team to set up the same sync for `dhis2-chap/chap-core`. Until then, a
maintainer pushes the source with `tx push -s` and adds PO files by hand.

The Modeling App must explicitly send the DHIS2 user's UI language, and the DHIS2
route must forward `Accept-Language`. Verify this through the deployed route as
well as directly against Chap; the browser's preferred language can differ from
the DHIS2 setting. Until a translated catalog is merged, responses remain English.
