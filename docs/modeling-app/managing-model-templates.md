# Getting new models into the modeling app

This document describes how new models can be added to the modeling app.

Note that when talking about adding a model to the modeling app, we are usually referring to adding a new **model template** (unconfigured model) that can then be used to create configured models (models that can be trained and used for predictions).


## Model seeding on startup

CHAP seeds the database with model templates and configured models every time the backend starts. The seeding process reads YAML files from `config/configured_models/` and is idempotent (existing models are updated, not duplicated).

### How seeding works

1. `default.yaml` is parsed first. For each model entry, only the **last listed version** is used -- earlier versions are kept as historical documentation.
2. All other `*.yaml` files in the directory (e.g. `local.yaml`, `benchmark_models.yaml`) are parsed next, with all versions available.
3. For each model, the seeding logic fetches `MLProject.yaml` from the GitHub repository at the specified commit to retrieve model metadata (name, description, required covariates, user options, etc.).
4. A model template is inserted (or updated) in the database, along with any named configurations that specify user option values and additional covariates.

### Adding custom models

Do not edit `default.yaml` directly -- it is overwritten on updates. Instead, create a new YAML file in `config/configured_models/` following this format:

```yaml
- url: https://github.com/org/model-repo
  versions:
    v1: "@<commit-sha-or-branch>"
  configurations:       # optional
    config_name:
      user_option_values:
        option_key: value
      additional_continuous_covariates:
        - rainfall
        - mean_temperature
```

See `config/configured_models/README.md` for full details on the file format and seeding logic.

Note that configured models can also be created directly through the modeling app (if the model template already exists).

We are currently also working on a system for adding new model templates through the modeling app. Note that this functionality has not been released yet, and is planned for a future release. The rest of this document describes how that system works.


## System for adding model templates through the modeling app

**This functionality is not released yet, but planned for a future release.**

The backend of chap-core provides API endpoints to manage model templates. In order to not allow arbitrary model templates to be added, a whitelist system is used.

### Whitelist System

#### How It Works

CHAP maintains a list of approved model templates via remote YAML whitelists. The backend:

1. Reads URLs from `config/approved_model_repos.yaml`
2. Fetches each URL and parses the YAML whitelist
3. Merges all results into a single approved list
4. Validates add requests against this merged list
5. Caches the whitelist for 5 minutes to reduce network calls

#### Whitelist Format

Remote whitelists use this format:

```yaml
- url: https://github.com/dhis2-chap/chap_auto_ewars
  versions:
    stable: "@209759add6e13778f7061b8add8fbe814799a6cb"
    nightly: "@main"

- url: https://github.com/dhis2-chap/ewars_template
  versions:
    v3: "@e4520a2123a3228c10947f2b25029c3f7190e320"
```

Each entry has:
- `url`: The base GitHub repository URL
- `versions`: Named versions mapping to Git refs (commits or branches)

See the API documentation for endpoints under `/api/v1/model-templates` for details on how to list and add model templates using this system.

### Adding a new model-template to the whitelist

In order to add a new model-template to our whitelist, you should make a pull-request to the [model-repositories](https://github.com/dhis2-chap/model-repositories) repository. Note that main.yaml is what is by default in the approved_model_repos.yaml file, so this is where one would usually add new model-templates if they are to be available to every deployment of chap-core.

It is also possible to add custom whitelist sources to your own deployment of chap-core, see the section below for deployers.

Specifically, follow these steps to add a new model to the approved list:

1. Fork the [model-repositories](https://github.com/dhis2-chap/model-repositories) repository
2. Edit `main.yaml` to add the new entry:
   ```yaml
   - url: https://github.com/your-org/your-model
     versions:
       latest: "@abc123def456"  # Use a specific commit hash for stability
   ```
3. Submit a pull request for review
4. Once merged, the model becomes available after the cache TTL (5 minutes) or backend restart

## For Deployers

### Custom Whitelist Sources

To add custom whitelist sources for your deployment:

1. Edit `config/approved_model_repos.yaml`:
   ```yaml
   - https://raw.githubusercontent.com/dhis2-chap/model-repositories/main/main.yaml
   - https://raw.githubusercontent.com/my-org/my-models/main/approved.yaml
   ```
2. Restart the backend for changes to take effect

### Security Considerations

- Only add whitelist URLs from trusted sources
- Prefer commit hashes over branch names for version pinning
- Regularly audit the approved model list
