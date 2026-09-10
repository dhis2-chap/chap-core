# 1. Installing Chap for model developers

In this guide, you'll install the Chap command-line tool. Once installed, you can run `chap eval` to test any model against real datasets — which you'll do in the next guide in this session.

**Reminder:** Windows users, use WSL (Windows Subsystem for Linux) as covered in [Prepare for installation](../external_models/prepare-for-installation.md).

## Installing Chap

Install Chap as a global tool using uv:

```bash
uv tool install chap-core --python 3.13
```

This installs the `chap` command-line tool globally, making it available from any directory.

## Installing and updating models

With Docker and Docker Compose v2 installed, use a model ID from the
[CHAP Model Marketplace](https://github.com/dhis2-chap/model-marketplace).
Both commands select `channels.stable` and require that version to be `verified`.
They do not select the `latest` channel or an unreviewed version. Marketplace
verification checks the model revision and chapkit service, not forecast quality.
Templates for model authors cannot be installed as forecasting models.

### For CLI evaluations

```bash
chap install chapkit_simple_multistep_model --local
chap update chapkit_simple_multistep_model --local
```

The model runs in a background container with a port bound to `127.0.0.1`.
The command prints its URL and an example `chap eval --model-name` argument;
use that URL with your dataset. No CHAP server is needed. The port may change
after an update, so use the URL printed by the latest command.
Local model settings are stored in `~/.chap/compose.models.yml`.

To stop local models:

```bash
docker compose -p chap-local-models -f ~/.chap/compose.models.yml stop
```

### For a CHAP deployment and the Modeling App

From the directory containing your running CHAP Compose deployment:

```bash
chap install chapkit_simple_multistep_model
chap update chapkit_simple_multistep_model
```

The service joins the deployment network and self-registers with CHAP, making it
available to the Modeling App. CHAP must already be running. If your deployment
uses different base files, supply them in the same order as when starting CHAP:

```bash
chap install chapkit_simple_multistep_model --compose-file compose.yml --compose-file compose.ghcr.yml
```

The commands create `compose.marketplace.yml` beside the first base file. Include
it in subsequent Docker Compose commands, for example
`docker compose -f compose.yml -f compose.marketplace.yml up -d`.
Continue using your deployment's existing `COMPOSE_PROJECT_NAME` and environment
settings. `SERVICEKIT_REGISTRATION_KEY` is forwarded when set in the environment
or deployment's `.env` file.

Only the selected model is pulled and started. Updates preserve its data volume
and Compose settings; failed updates attempt to restart the previous image.
Use `--platform linux/amd64` for models that only publish AMD64 images, such as
R-INLA models on Apple Silicon. The platform is retained for subsequent updates.

### Custom chapkit models

Custom images must implement the chapkit service API on port 8000 and, for a CHAP
deployment, support chapkit self-registration. They are not marketplace-reviewed.
You accept responsibility for running their code, sharing data with them, and
using their forecasts. Both installation and updates require `--accept-risk`:

```bash
chap install my_model --local --image ghcr.io/my-org/my-model:v1 --accept-risk
chap update my_model --local --image ghcr.io/my-org/my-model:v2 --accept-risk
```

Omit `--local` to add the custom service to your CHAP deployment. Updating a custom
model without `--image` pulls its existing image reference again; it never
switches to a marketplace model automatically. Prefer version tags or digests
for reproducible custom installations.

## Exercise

### Verify your installation

Run the following command:

```bash
chap --help
```

You should see output listing available commands including `eval`, `plot-backtest`, and `export-metrics`.

**Verification:** If you see the help output with available commands, Chap is installed correctly. You're ready for the next guide: [Implement your own model from a minimalist example](../kigali-workshop/kigali-webinar-series/session-3/fork-example.md).
