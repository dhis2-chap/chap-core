# 1. Installing Chap for model developers

In this guide, you'll install the Chap command-line tool. Once installed, you can run `chap eval` to test any model against real datasets — which you'll do in the next guide in this session.

**Reminder:** Windows users, use WSL (Windows Subsystem for Linux) as covered in [Prepare for installation](../external_models/prepare-for-installation.md).

## Installing Chap

Install Chap as a global tool using uv:

```bash
uv tool install chap-core --python 3.13
```

This installs the `chap` command-line tool globally, making it available from any directory.

## Running marketplace models

With Docker and Docker Compose v2 installed, use a model ID from the
[CHAP Model Marketplace](https://github.com/dhis2-chap/model-marketplace).
The commands below select `channels.stable` and require that version to be
`verified`. They do not select the `latest` channel or an unreviewed version.
Marketplace verification checks the model revision and chapkit service, not
forecast quality. Templates for model authors cannot be run as forecasting models.

### For CLI evaluations

```bash
chap model start chapkit_simple_multistep_model
```

The model runs in a background container with a port bound to `127.0.0.1`.
The command prints its URL and an example `chap eval --model-name` argument;
use that URL with your dataset. No CHAP server is needed. Running the command
again moves an already started model to the current verified version. The port
may change after that, so use the URL printed by the latest command.
Local model settings are stored in `~/.chap/compose.models.yml`.

To stop a local model:

```bash
chap model stop chapkit_simple_multistep_model
```

Its data volume is kept so a later start resumes from it; pass `--delete-data`
to remove it permanently.

### For a CHAP deployment and the Modeling App

Installing a model into a deployment is done with `chap-admin`, which comes with
the same package and talks to the running CHAP instance over its REST API. From
the directory containing your running CHAP Compose deployment:

```bash
chap-admin install chapkit_simple_multistep_model
chap-admin update chapkit_simple_multistep_model
```

`chap-admin` reaches CHAP at `http://localhost:8000` unless `CHAP_URL` or
`--url` says otherwise, and sends `CHAP_API_TOKEN` or `--token` when the
deployment requires a token. It reads both from the environment, not from a
deployment's `.env` file.

Installing does three things, in this order, and each step can be repeated if a
later one fails:

1. The model service is written into `compose.marketplace.yml` beside the first
   base file, pinned to the verified image.
2. The model template and its verified configurations are registered in CHAP
   from the marketplace entry, so the model shows up in the Modeling App with
   the reviewed ways to run it. The service does not need to be running for this.
3. The service is pulled and started on the deployment network. Pass
   `--no-start` to skip this, for example to start every model later with one
   `docker compose up`.

Updating registers the new version as a new template version with its
configurations. Earlier versions and the evaluations made with them are untouched.

If your deployment uses different base files, supply them in the same order as
when starting CHAP:

```bash
chap-admin install chapkit_simple_multistep_model --compose-file compose.yml --compose-file compose.ghcr.yml
```

Because the commands name the base files explicitly, Docker Compose does not load
`compose.override.yml` on its own. List it with `--compose-file` as well if your
deployment uses one.

Include `compose.marketplace.yml` in subsequent Docker Compose commands, for
example `docker compose -f compose.yml -f compose.marketplace.yml up -d`.
Continue using your deployment's existing `COMPOSE_PROJECT_NAME` and environment
settings. `SERVICEKIT_REGISTRATION_KEY` is forwarded when set in the environment
or deployment's `.env` file.

Only the selected model is pulled and started. Updates preserve its data volume
and Compose settings; failed updates attempt to restart the previous image.
Use `--platform linux/amd64` for models that only publish AMD64 images, such as
R-INLA models on Apple Silicon. The platform is retained for subsequent updates.

A deployment can also declare its marketplace models in a seed file under
`config/configured_models/` with a `marketplace:` entry; CHAP then registers
them itself at startup. See the README in that directory.

### Removing a model

```bash
chap-admin uninstall chapkit_simple_multistep_model
```

The model template and its configured models are retired in CHAP so they leave
the pickers; they are never deleted, since evaluations reference them. The
service is then stopped and removed. The model's data volume is kept so a later
install resumes from it; pass `--delete-data` to remove it permanently.
Uninstalling the last model leaves `compose.marketplace.yml` in place with no
services, so you can keep passing it to Docker Compose.

### A different model registry

Set `CHAP_MARKETPLACE_URL` in your shell to resolve models from another registry,
such as one hosting your organisation's own models. The commands read it
from the environment, not from a deployment's `.env` file. Its models are not
marketplace-reviewed, so installation and updates require `--accept-risk`:

```bash
export CHAP_MARKETPLACE_URL=https://models.example.org/registry
chap-admin install my_org_model --accept-risk
```

### Custom chapkit models

Custom images must implement the chapkit service API on port 8000 and, for a CHAP
deployment, support chapkit self-registration. They are not marketplace-reviewed.
You accept responsibility for running their code, sharing data with them, and
using their forecasts. Both installation and updates require `--accept-risk`:

```bash
chap model start my_model --image ghcr.io/my-org/my-model:v1 --accept-risk
chap-admin install my_model --image ghcr.io/my-org/my-model:v1 --accept-risk
chap-admin update my_model --image ghcr.io/my-org/my-model:v2 --accept-risk
```

A custom image has no marketplace entry to register the model from, so
`chap-admin` starts the service first, waits for it to register with CHAP, and
then stores the template from the service's own description with one default
configuration. `--no-start` is therefore not available for custom images.
Updating a custom model without `--image` pulls its existing image reference
again; it never switches to a marketplace model automatically. Prefer version
tags or digests for reproducible custom installations.

## Exercise

### Verify your installation

Run the following command:

```bash
chap --help
```

You should see output listing available commands including `eval`, `plot-backtest`, and `export-metrics`.

**Verification:** If you see the help output with available commands, Chap is installed correctly. You're ready for the next guide: [Implement your own model from a minimalist example](../kigali-workshop/kigali-webinar-series/session-3/fork-example.md).
