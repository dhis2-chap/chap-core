# Translating API text

Scoring metric names, descriptions and comparison descriptions are translated in
the metric catalogue and benchmark leaderboard responses. Clients send the user's
UI language in `Accept-Language`, for example `fr` or `fr-CA`. Chap honors quality
weights and falls back to a base language, then English. Missing and fuzzy
translations also fall back to the English source string. IDs and scores stay the
same in every language.

Model names, plot text, outbreak metrics and error messages are outside this first
pass. French translations and HISP WCA review are tracked in CLIM-1895.

## Adding strings

Wrap metric metadata in `N_("English text")` imported from `chap_core.i18n`.
This marks it for Babel without translating shared specifications at import time.
The REST endpoints translate the strings when building their responses.

`make lint` regenerates `chap_core/locales/en.pot`; commit it with the code change.
`make check` rejects stale catalogs in CI. To extract without running other checks:

<!-- Cannot test: rewrites the committed source catalog. -->
```console
make extract-translations
```

Translation files live at `chap_core/locales/<language>.po`. They ship inside the
Python package and are compiled in memory once per process, so deployments need no
separate compilation step. Restart Chap after updating catalogs.

## Connecting Transifex (one-time setup)

The intended project is `hisp-uio/modeling-app`, shared with chap-frontend, with a
separate resource for chap-core. The files alone do not enable remote syncing.
A project maintainer must:

1. Give the Transifex GitHub integration access to `dhis2-chap/chap-core`.
2. In the project's GitHub integration, link this repository's `master` branch
   using `transifex.yml`. Keep the existing frontend connection.
3. Select automatic translation pull requests and the team's review threshold.
4. Confirm the generated resource slug and update the resource section in
   `.tx/config` if it differs from `chap-core-en-pot`. That file configures the
   optional Transifex CLI; `transifex.yml` configures the GitHub integration.
5. Verify a source update reaches Transifex and a translation update opens a PR.

See the [Transifex GitHub setup guide](https://help.transifex.com/en/articles/6265125-github-installation-and-configuration).
Once connected, committed source catalog updates sync automatically and translation
PRs bring back PO files. Use the normal PR template and `Review needed:` line when
reviewing those PRs.

The Modeling App must explicitly send the DHIS2 user's UI language, and the DHIS2
route must forward `Accept-Language`. Verify this through the deployed route as
well as directly against Chap; the browser's preferred language can differ from
the DHIS2 setting. Until a translated catalog is merged, responses remain English.
