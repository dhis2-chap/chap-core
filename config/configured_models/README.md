# Configured Models Seeding

This directory contains YAML files that define which model templates and configured models are seeded into the database on startup.

## How it works

On startup, the REST API calls `seed_configured_models_from_config_dir()` (in `chap_core/database/model_template_seed.py`), which:

1. Parses `default.yaml` first, keeping only the **last version** listed for each model (earlier versions serve as historical documentation).
2. Parses all other `*.yaml` files in this directory (e.g. `local.yaml`, `benchmark_models.yaml`). These keep all versions through parsing, but seeding still only uses the last one, so listing several has no effect.
3. For each model entry, takes the last version and constructs a GitHub URL (`{url}@{commit}`).
4. Fetches the `MLProject.yaml` from the GitHub repository at that commit to get model metadata (name, description, covariates, user options, etc.).
5. Inserts a `ModelTemplateDB` row for the `(name, version)` pair, unless that pair is already stored.
6. For each configuration listed under `configurations:`, inserts a `ConfiguredModelDB` row with the specified user option values and additional covariates.
7. Finally, adds a built-in naive model template used for testing.

Seeding is idempotent: a `(name, version)` pair that is already stored is reused, as long as it still
points at the same commit.

## Versions are full commit shas

Every value under `versions:` must be a full 40-character commit sha, with or without a leading `@`.
Branches, tags and short shas are rejected when the file is parsed, so CHAP fails to start with a message
naming the file, the label and the value. This holds for every label in the file, not only the last one.
A label pins one revision, which is what makes evaluations against it reproducible and comparable.

## Versions are write-once

A stored version is never rewritten, because backtests and predictions point at the row and must keep
describing the code they ran.

- **To publish new model code, add a new version entry.** The new row becomes the live one for that name,
  and the old row stays in the database so its backtests keep their true provenance.
- **Reusing a label with a new sha fails startup.** The stored row keeps its commit and CHAP refuses to
  start, with a message naming the label, the stored sha and the new sha. Add a new label instead.
- **Editing a template under a version that is already seeded has no effect.** CHAP keeps the stored row and
  logs a warning that names the fields it ignored.
- **Editing `configurations:` does take effect.** A changed `user_option_values` or
  `additional_continuous_covariates` adds a new configured model that becomes the live one for that name.
  The old configured model stays in the database for the backtests that used it.

## File format

```yaml
- url: https://github.com/org/model-repo
  versions:
    v1: "@<commit-sha>"           # historical documentation only
    v2: "@<commit-sha>"           # last entry is the one that gets seeded
  configurations:                 # optional, defaults to a single "default" config
    config_name:
      user_option_values:
        option_key: value
      additional_continuous_covariates:
        - rainfall
        - mean_temperature
```

### Fields

- **url**: The GitHub repository URL for the model.
- **name** (optional): Overrides the template name declared by the model itself. Use it to avoid name clashes when seeding two variants of the same model.
- **versions** (required): Named versions mapping to full commit shas, optionally prefixed with `@`. Branches and tags are not allowed. Only the last entry is seeded, in every file -- earlier entries serve as historical documentation.
- **configurations** (optional): Named configurations for the model template. Each configuration can set `user_option_values` (model-specific parameters) and `additional_continuous_covariates`. If omitted, a single "default" configuration with empty values is created.

### Marketplace models

A chapkit model from the [CHAP Model Marketplace](https://github.com/dhis2-chap/model-marketplace) is
declared with a single `marketplace:` entry naming its marketplace id:

```yaml
- marketplace: chapkit_ewars_model
```

On startup CHAP resolves the model's verified stable version from the registry and stores its template
and verified configurations, the same registration `chap-admin install` makes over the REST API. The entry
does not run the service: start it with `chap-admin install` or an overlay such as `compose.ewars.yml`.
The template's source address is the in-network name `chap-admin install` uses,
`http://marketplace-<service-id>:8000`; a service started under another name is found through its
self-registration instead. A chapkit service that only self-registers is not a model in CHAP; see
[Running Your Own Model](../../docs/modeling-app/running-your-own-model.md) for images that have no
marketplace entry.

## Adding models

Do not edit `default.yaml` directly -- it is overwritten on updates. Instead, create a new `*.yaml` file (e.g. `local.yaml`) following the same format. Any `*.yaml` file in this directory (except files ending in `.disabled`) will be read on startup.

## Key source files

- `chap_core/models/local_configuration.py` -- YAML parsing logic
- `chap_core/database/model_template_seed.py` -- database seeding logic
- `chap_core/database/database.py` (`create_db_and_tables`) -- startup entry point that triggers seeding
