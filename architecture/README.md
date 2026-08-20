# CHAP architecture model

A C4 model of the CHAP platform, written as code in [`workspace.dsl`](workspace.dsl)
(Structurizr DSL). It is aimed at devops and power users who run, deploy or
operate CHAP rather than at people reading the source.

The model is the single source of truth for the diagrams. Edit `workspace.dsl`
and the views regenerate; there are no hand-drawn images to keep in sync.

## The levels

The model is layered so no single diagram tries to show everything. Open them
in this order:

1. **L1 - System landscape** (`L1_Landscape`) - the whole landscape in one
   picture: DHIS2 `<=>` CHAP Core `=>` model services, with every actor. Shows
   who uses what and where data crosses a boundary.
2. **L2 - Containers** - the running pieces inside a system:
   - `L2_ChapCore` - REST API, Celery worker, Redis/Valkey, PostgreSQL, CLI.
   - `L2_Chapkit` - a chapkit model service: Service API, SQLite store, web
     console.
3. **L3 - Components** - drill-down into the non-trivial containers:
   - `L3_CoreAPI` - CHAP Core REST API (v1/v2 routers, Orchestrator).
   - `L3_CoreWorker` - CHAP Core worker (worker functions, runners, chapkit client).
   - `L3_ChapkitService` - chapkit Service API internals (ML/config/artifact/job
     routers, ML manager, async scheduler, pluggable model runner, registration).
4. **Flows** - dynamic views for the three journeys that matter operationally:
   ingest a dataset, run an evaluation (backtest), and run a prediction whose
   forecasts end up back in DHIS2.

## The actors (conceptual)

| Actor | What it is | Key point for operators |
| --- | --- | --- |
| **DHIS2** | External health information system. | Source of case/climate/org-unit data and the final destination for forecasts. CHAP never calls DHIS2 directly. |
| **CHAP Modelling App** (`chap-frontend`) | DHIS2 app, the main client. | Submits work to the CHAP API, polls jobs, pulls forecasts, and writes them back into DHIS2. The DHIS2 write happens here, not in CHAP Core. |
| **REST API** | FastAPI process in CHAP Core. | Validates input, enqueues jobs, serves results. Stateless: scale horizontally. |
| **Celery worker** | Worker process in CHAP Core. | Does the heavy work (harmonise, backtest, predict). Scale by adding worker replicas behind the same Redis. |
| **Redis / Valkey** | Broker + state. | Celery broker, job metadata, and the live chapkit service registry. |
| **PostgreSQL** | Database. | Datasets, model templates/configs, backtests, predictions. |
| **chapkit model services [0..*]** | Self-contained model services - **one per model**, zero-to-many running (framework + runtime; own repo `chap-sdk/chapkit`). The now-preferred path over MLproject. | Not just "an MLproject in a repo": each service is a FastAPI app with its own SQLite store, async job scheduler, typed config/artifact storage and a web console. Each registers with CHAP Core and is called by the worker for train/predict. Run and scale independently. See `L2_Chapkit` / `L3_ChapkitService`. |
| **Model source repos** | Git / MLproject, **one per model**. | MLproject models the worker clones and runs in-process via a runner. |
| **CHAP CLI** | Local entry point. | Lets a model developer run/evaluate models without the API or DHIS2. |

A few directional facts the diagrams encode (and that are easy to get wrong):

- Predictions are **collected and stored in CHAP**, not auto-pushed. CHAP never
  writes to DHIS2 itself. The Modelling App pulls the quantiles from the API for
  **human review**, and only **after approval** does it (optionally) write the
  `dataValueSets` into DHIS2.
- Recurring predictions are described by a `PredictionSetup` stored in CHAP (its
  cron expression and quantile-to-data-element mapping); the trigger that fires
  them on schedule is out of scope for this model.
- Model execution is either **remote** (chapkit HTTP services) or **in-process**
  (a runner cloning a model repo). Both are driven by the worker.
- A model developer works locally with the CHAP CLI or chapkit, but when
  **publishing** a model picks one of two packagings: **Option A** - an
  MLproject repo (cloned and run in-process by the worker), or **Option B** - a
  chapkit service (called over HTTP). The two consumption paths in the diagrams
  are the two ends of that choice. Concrete examples of both live in the
  [chap-models org](https://github.com/orgs/chap-models/repositories) - e.g.
  `chap_auto_ewars` (MLproject) and `chapkit_ewars_model` (chapkit), across
  Python, R and other languages.

## Reading the diagrams

In the **interactive viewer** (`make architecture`) and its PNG export, shape,
colour and logo all carry meaning, so you can tell what a box is at a glance:

- **Cylinder** = a datastore (PostgreSQL, the Redis/Valkey broker+store, the
  chapkit SQLite store). Logos tell same-shaped stores apart.
- **Person** = a human role; plain boxes are systems/containers/components.
- Technology **logos** (PostgreSQL, Redis, SQLite, FastAPI, React) are shown on
  the relevant containers. They are fetched from a CDN at render time, so the
  interactive viewer and the export need network access; offline, the boxes
  still render, just without the logo.

The committed [Mermaid docs page](../docs/contributor/architecture_model.md)
deliberately carries fewer of these cues: **cylinders survive**, but the C4
colours, the person shape and the technology logos do not - the palette is
dropped so the diagrams follow the MkDocs light/dark theme. Every box there is
labelled with its type (`[Person]`, `[Software System]`, `[Container: …]`,
`[Component]`) instead, so nothing is ambiguous - it is just plainer. Use the
interactive viewer when you want the full visual encoding.

## Viewing and editing the diagrams

The diagrams render from `workspace.dsl`. The interactive viewer gives you
zoom, pan, fullscreen and click-through between levels - which the static
Mermaid diagrams in the mkdocs docs do not.

Run the viewer locally (no account needed), from the repo root:

```bash
make architecture          # serves http://localhost:6080
```

Then open <http://localhost:6080>. Edit `workspace.dsl` and refresh the browser
to see changes.

To validate the DSL before committing or in CI:

```bash
make architecture-validate
```

## Exporting PNGs and pre-warming thumbnails

```bash
make architecture-export      # needs port 6080 free
```

This renders every view to `architecture/diagrams/<ViewKey>.png`. The output is
**gitignored, not committed** - the viewable-in-the-repo copy of the model is the
Mermaid docs page (see below), which stays in sync with `workspace.dsl` without
carrying binaries. The target is self-contained: it starts a temporary Structurizr
instance, drives a headless browser over each view via Structurizr's diagram
scripting API, writes the PNGs, and tears the instance down.

The prebuilt `structurizr/structurizr` image cannot export PNG/SVG itself
("not supported in this build"), so the export uses the official Playwright
Docker image. The script that does the work is
[`export-diagrams.js`](export-diagrams.js).

As a side effect, visiting every view also populates Structurizr's own thumbnail
cache (`architecture/.structurizr/1/images/*-thumbnail.png`, gitignored). Those
persist, so after one `make architecture-export` the diagram-finder thumbnails
show immediately in later `make architecture` sessions instead of rendering
lazily on first click.

The renderer toolchain is **version pinned** so a re-export does not change the
output without a source change: the Structurizr image, the Playwright image,
mermaid-cli and PlantUML are all pinned to explicit versions in the `Makefile`.

## Trying other renderers

Structurizr DSL stays the single source of truth, but the same model can be
re-rendered by other tools if you want to compare locally:

```bash
make architecture-export-mermaid    # -> architecture/diagrams/mermaid/*.png
make architecture-export-plantuml   # -> architecture/diagrams/plantuml/*.png
```

Both write under `architecture/diagrams/<renderer>/`, which is gitignored.

- **Mermaid** and **C4-PlantUML** are derived automatically from `workspace.dsl`
  (`structurizr export -format …`) and rendered to PNG. Note: neither carries the
  technology logos.
- **D2** and **Ilograph** are not supported by this Structurizr build's exporter.
- **LikeC4** was evaluated and dropped: it cannot consume `workspace.dsl`, so it
  needed a hand-maintained second copy of the whole model that silently drifted
  from the source of truth.

## Publishing the model into the docs site

```bash
make architecture-export-docs   # -> docs/contributor/architecture_model.md
```

This regenerates a contributor docs page with every view as native, theme-aware
Mermaid (the Structurizr Mermaid export's HTML labels are collapsed to plain text
by [`mermaid_to_docs.py`](mermaid_to_docs.py)). mkdocs renders those fences
natively via `pymdownx.superfences`, so the build needs no Docker and no image
files. **This page is the committed, reviewable rendering of the model** - rerun
the target and commit it whenever `workspace.dsl` changes.

CI enforces that. `make architecture-check-docs` regenerates the page and fails if
the result differs from what is committed, and the docs workflow runs it on every
pull request. The export is byte-reproducible (no hidden state in the gitignored
`workspace.json`, and identical output on arm64 and amd64), so a diff there always
means the page is stale - not that the renderer drifted.

## A note on Structurizr licensing

Structurizr consolidated its tooling ("vNext"). The functionality we use -
`local` (viewer), `validate`, and `export` - **remains free and open source**.
Only the on-prem multi-user *server* (auth, Elasticsearch) needs a paid license
via prebuilt binaries (free if built from source); that is not used here. The
one thing that affected us was the cloud-hosted theme EOL (30 Sep 2026), already
removed in favour of explicit styles.

## Under the hood

The make targets are thin wrappers around the Structurizr Docker image:

```bash
# make architecture
docker run -it --rm -p 6080:8080 \
  -v "$(pwd)/architecture:/usr/local/structurizr" \
  structurizr/structurizr:2026.05.22 local

# make architecture-validate
docker run --rm -v "$(pwd)/architecture:/work" -w /work \
  structurizr/structurizr:2026.05.22 validate -workspace workspace.dsl
```

> Note: the older `structurizr/lite` and `structurizr/cli` images are retired
> and now only print a migration notice. Use `structurizr/structurizr` as above.
> The image tag is pinned (see the version-pinning note above); bump it
> deliberately and re-export in the same commit.
