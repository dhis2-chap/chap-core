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

Shapes and logos carry meaning, so you can tell what a box is at a glance:

- **Cylinder** = a datastore (PostgreSQL, the Redis/Valkey broker+store, the
  chapkit SQLite store). Logos tell same-shaped stores apart.
- **Person** = a human role; plain boxes are systems/containers/components.
- Technology **logos** (PostgreSQL, Redis, SQLite, FastAPI, React) are shown on
  the relevant containers. They are fetched from a CDN at render time, so the
  interactive viewer and the export need network access; offline, the boxes
  still render, just without the logo.

## Viewing and editing the diagrams

The diagrams render from `workspace.dsl`. The interactive viewer gives you
zoom, pan, fullscreen and click-through between levels - which the static
Mermaid diagrams in the mkdocs docs do not.

Run the viewer locally (no account needed), from the repo root:

```bash
make architecture          # serves http://localhost:8080
```

Then open <http://localhost:8080>. Edit `workspace.dsl` and refresh the browser
to see changes.

To validate the DSL before committing or in CI:

```bash
make architecture-validate
```

## Exporting PNGs and pre-warming thumbnails

```bash
make architecture-export      # needs port 8080 free
```

This renders every view to `architecture/diagrams/<ViewKey>.png` (committed, so
the diagrams are viewable in the repo without running anything). The target is
self-contained: it starts a temporary Structurizr instance, drives a headless
browser over each view via Structurizr's diagram scripting API, writes the PNGs,
and tears the instance down.

The prebuilt `structurizr/structurizr` image cannot export PNG/SVG itself
("not supported in this build"), so the export uses the official Playwright
Docker image. The script that does the work is
[`export-diagrams.js`](export-diagrams.js).

As a side effect, visiting every view also populates Structurizr's own thumbnail
cache (`architecture/.structurizr/1/images/*-thumbnail.png`, gitignored). Those
persist, so after one `make architecture-export` the diagram-finder thumbnails
show immediately in later `make architecture` sessions instead of rendering
lazily on first click.

## Trying other renderers

Structurizr DSL stays the single source of truth, but the same model can be
re-rendered by other tools so you can compare. These are experimental:

```bash
make architecture-export-mermaid    # -> architecture/diagrams/mermaid/*.png
make architecture-export-plantuml   # -> architecture/diagrams/plantuml/*.png
make architecture-export-likec4     # -> architecture/diagrams/likec4/*.png
make architecture-likec4            # interactive LikeC4 viewer at :5180
```

Each renderer's PNGs sit next to the Structurizr ones under
`architecture/diagrams/<renderer>/`, so you can open the same view across folders
for side-by-side comparison.

- **Mermaid** and **C4-PlantUML** are derived automatically from `workspace.dsl`
  (`structurizr export -format …`) and rendered to PNG. Note: neither carries the
  technology logos.
- **LikeC4** is a separate, hand-written model ([`likec4/chap.likec4`](likec4/chap.likec4))
  kept in sync manually - it is *not* derived from `workspace.dsl`. Its strength
  is the interactive viewer (`make architecture-likec4`). LikeC4's own headless
  `export png` fails in Docker ("Failed N of N views"), so the export target
  instead builds the static site, serves it, and screenshots each view via
  Playwright ([`export-likec4.js`](export-likec4.js)). It needs Graphviz + a
  browser, so it is slower than the other two.
- **D2** and **Ilograph** are not supported by this Structurizr build's exporter.

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
docker run -it --rm -p 8080:8080 \
  -v "$(pwd)/architecture:/usr/local/structurizr" \
  structurizr/structurizr local

# make architecture-validate
docker run --rm -v "$(pwd)/architecture:/work" -w /work \
  structurizr/structurizr validate -workspace workspace.dsl
```

> Note: the older `structurizr/lite` and `structurizr/cli` images are retired
> and now only print a migration notice. Use `structurizr/structurizr` as above.
