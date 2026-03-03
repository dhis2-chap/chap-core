# Configured Models Seeding

This directory contains YAML files that define which model templates and configured models are seeded into the database on startup.

## How it works

On startup, the REST API calls `seed_configured_models_from_config_dir()` (in `chap_core/database/model_template_seed.py`), which:

1. Parses `default.yaml` first, keeping only the **last version** listed for each model (earlier versions serve as historical documentation).
2. Parses all other `*.yaml` files in this directory (e.g. `local.yaml`, `benchmark_models.yaml`), keeping **all versions** listed.
3. For each model entry, takes the last version and constructs a GitHub URL (`{url}@{commit}`).
4. Fetches the `MLProject.yaml` from the GitHub repository at that commit to get model metadata (name, description, covariates, user options, etc.).
5. Inserts a `ModelTemplateDB` row (or updates it if the name already exists).
6. For each configuration listed under `configurations:`, inserts a `ConfiguredModelDB` row with the specified user option values and additional covariates.
7. Finally, adds a built-in naive model template used for testing.

Models that already exist in the database (matched by name) are updated rather than duplicated, so seeding is idempotent.

## File format

```yaml
- url: https://github.com/org/model-repo
  versions:
    v1: "@<commit-sha>"          # historical, ignored in default.yaml
    v2: "@<commit-sha-or-branch>" # last entry is the one that gets seeded
  configurations:                 # optional, defaults to a single "default" config
    config_name:
      user_option_values:
        option_key: value
      additional_continuous_covariates:
        - rainfall
        - mean_temperature
```

### Fields

- **url**: GitHub repository URL for the model.
- **versions**: Named versions mapping to git refs. Prefix with `@` for commits/branches. In `default.yaml`, only the last version is used; in other files, all versions are available.
- **configurations** (optional): Named configurations for the model template. Each configuration can set `user_option_values` (model-specific parameters) and `additional_continuous_covariates`. If omitted, a single "default" configuration with empty values is created.

## Adding models

Do not edit `default.yaml` directly -- it is overwritten on updates. Instead, create a new `*.yaml` file (e.g. `local.yaml`) following the same format. Any `*.yaml` file in this directory (except files ending in `.disabled`) will be read on startup.

## Key source files

- `chap_core/models/local_configuration.py` -- YAML parsing logic
- `chap_core/database/model_template_seed.py` -- database seeding logic
- `chap_core/database/database.py` (`create_db_and_tables`) -- startup entry point that triggers seeding
