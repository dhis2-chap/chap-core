workspace "CHAP" "Architecture model for the CHAP climate-and-health platform: DHIS2 <=> chap-core => worker / chapkit model services." {

    !identifiers hierarchical

    model {
        !impliedRelationships true

        # --- People -------------------------------------------------------
        analyst = person "Implementer / Analyst" "Configures models and reviews forecasts inside the DHIS2 Modelling App."
        modeller = person "Model developer" "Develops models locally (CHAP CLI or chapkit), then publishes them one of two ways: an MLproject repo or a chapkit service."

        # --- External systems --------------------------------------------
        dhis2 = softwareSystem "DHIS2" "Health information system. Source of case, climate and org-unit data; destination for forecast data values." {
            tags "External"
        }

        modellingApp = softwareSystem "CHAP Modelling App (chap-frontend)" "Embedded DHIS2 app. The primary client of CHAP Core and the component that writes forecasts back into DHIS2." {
            tags "External"
        }

        modelRepos = softwareSystem "Model source repos" "Git repositories / MLproject definitions - one per model - run in-process by CHAP Core." {
            tags "External"
        }

        # --- CHAP Core ----------------------------------------------------
        chapCore = softwareSystem "CHAP Core" "Climate-and-health modelling backend: ingests data, runs evaluations and predictions, and serves results." {

            api = container "REST API" "Serves the v1/v2 HTTP API, validates input, enqueues long-running jobs and serves results." "FastAPI / Uvicorn (Python)" {
                tags "FastAPI"
                common = component "Common routes" "Health, readiness and system-info endpoints."
                v1 = component "v1 routers" "crud, analytics, jobs, visualization. Datasets, backtests, predictions, job polling."
                v2 = component "v2 routers" "Service registry (chapkit self-registration) plus a read-only reverse proxy to live chapkit services."
                orchestrator = component "Orchestrator" "Redis-backed TTL registry of live chapkit services; other components read the live set from it."
            }

            worker = container "Celery worker" "Consumes queued jobs and runs dataset harmonisation, backtests and predictions." "Celery (Python)" {
                dbfns = component "Worker functions" "db_worker_functions: harmonise datasets, run backtests, run predictions; read/write the database."
                runners = component "TrainPredict runners" "Run MLproject models in-process: Docker / UV / Conda / Renv / MLflow / CLI."
                chapkitClient = component "Chapkit REST client" "CHAPKitRestAPIWrapper: httpx client calling remote chapkit model services."
            }

            cli = container "CHAP CLI" "Local entry point for running and evaluating models without the API." "Cyclopts (Python)"

            redis = container "Redis / Valkey" "Celery broker and result backend, job metadata (job_meta) and chapkit service registry." "Valkey 8" {
                tags "Database,Redis"
            }

            db = container "PostgreSQL" "Datasets, observations, model templates/configs, backtests, predictions." "PostgreSQL 17" {
                tags "Database,PostgreSQL"
            }
        }

        # --- chapkit model service (own repo: chap-sdk/chapkit) -----------
        chapkit = softwareSystem "chapkit model services [0..*]" "Self-contained model services - one per model, the now-preferred path. Each exposes the standard CHAP train/predict contract over HTTP and registers itself with CHAP Core." {

            serviceApi = container "Service API" "FastAPI app assembled by chapkit's MLServiceBuilder; implements the standard train/predict/config/artifact/job REST contract." "FastAPI (Python)" {
                tags "FastAPI"
                registration = component "Registration & health" "Self-registers with CHAP Core and sends heartbeats; serves /health and /api/v1/info (service identity CHAP Core reads)."
                mlRouter = component "ML router" "/api/v1/ml: $train, $predict, $validate, $generate-sample-data."
                configRouter = component "Config router" "/api/v1/configs: typed, Pydantic-validated model configuration CRUD."
                artifactRouter = component "Artifact router" "/api/v1/artifacts: artifact CRUD - tree, expand, metadata, linked config, download (trained models, predictions)."
                jobsRouter = component "Jobs router" "/api/v1/jobs: async job status and cancellation."
                mlManager = component "ML manager" "Train/predict pipelines; turns runner output into typed, versioned artifacts."
                jobScheduler = component "Job scheduler" "In-memory async scheduler; runs train/predict as ULID-tracked background jobs."
                modelRunner = component "Model runner" "Pluggable train/predict implementation: functional, class-based, or shell (Python / R)."
                store = component "Artifact & config store" "Trained-model artifacts, predictions and configs; tree-structured and Alembic-migrated. Embedded in-process (same service)." "SQLite" {
                    tags "Database,SQLite"
                }
            }

            console = container "Web console" "Built-in SPA to browse configs/artifacts/jobs and trigger train/predict against the service." "React SPA" {
                tags "React"
            }
        }

        # --- Relationships: people & external systems --------------------
        analyst -> modellingApp "Configures models, reviews forecasts"
        modeller -> chapCore.cli "Develops & evaluates models locally"
        modeller -> modelRepos "Publishes a model -- Option A: MLproject repo"
        modeller -> chapkit.console "Develops & tests a model -- Option B: chapkit service"
        modeller -> chapkit.serviceApi "Publishes (deploys the self-registering service)"

        modellingApp -> dhis2 "Reads case/climate/org-unit data; after review, optionally writes approved forecast data values" "DHIS2 Web API"
        modellingApp -> chapCore.api.v1 "Submits data, runs evaluations/predictions, polls jobs, pulls forecasts" "HTTPS/JSON (OpenAPI client)"

        # --- Relationships: CHAP Core API components ---------------------
        chapCore.api.v2 -> chapCore.api.orchestrator "Registers / lists services"
        chapCore.api.v1 -> chapCore.redis "Queues jobs; reads job status"
        chapCore.api.v1 -> chapCore.db "Reads/writes datasets, models, results"
        chapCore.api.orchestrator -> chapCore.redis "Live service registry (TTL keys)"
        chapCore.api.v1 -> chapCore.api.orchestrator "Reads live services to sync model templates"

        # --- Relationships: CHAP Core worker components ------------------
        chapCore.worker.dbfns -> chapCore.redis "Job lifecycle: fetch job, write job_meta (via Celery task wrapper)"
        chapCore.worker.dbfns -> chapCore.db "Reads datasets/models; writes forecasts & metrics"
        chapCore.worker.dbfns -> chapCore.redis "Resolves live service URL from registry (TTL keys)"
        chapCore.worker.dbfns -> chapCore.worker.runners "Runs in-process models"
        chapCore.worker.dbfns -> chapCore.worker.chapkitClient "Calls remote models"
        chapCore.worker.runners -> modelRepos "Clones & runs model code" "git / Docker / MLflow / UV"

        # --- Relationships: CHAP Core CLI --------------------------------
        chapCore.cli -> modelRepos "Runs model code locally"

        # --- Relationships: chapkit <-> CHAP Core ------------------------
        chapCore.worker.chapkitClient -> chapkit.serviceApi.mlRouter "Trains & predicts" "HTTP $train / $predict"
        chapkit.serviceApi.registration -> chapCore.api.v2 "Registers & sends heartbeats" "HTTP $register / $ping"
        chapCore.api.v2 -> chapkit.serviceApi "Read-only proxy to live service (artifacts/configs/jobs)" "HTTP GET/HEAD"

        # --- Relationships: chapkit internals ----------------------------
        chapkit.serviceApi.mlRouter -> chapkit.serviceApi.mlManager "Submits train/predict requests"
        chapkit.serviceApi.mlManager -> chapkit.serviceApi.jobScheduler "Schedules background job"
        chapkit.serviceApi.jobScheduler -> chapkit.serviceApi.modelRunner "Runs train / predict"
        chapkit.serviceApi.mlManager -> chapkit.serviceApi.store "Reads/writes artifacts & configs"
        chapkit.serviceApi.configRouter -> chapkit.serviceApi.store "Reads/writes configs"
        chapkit.serviceApi.artifactRouter -> chapkit.serviceApi.store "Reads artifact tree"
        chapkit.serviceApi.jobsRouter -> chapkit.serviceApi.jobScheduler "Reads job status"
        chapkit.console -> chapkit.serviceApi "Browses configs/artifacts/jobs; triggers train/predict" "REST"
    }

    views {
        systemLandscape "L1_Landscape" "Level 1 - System landscape. Every actor: DHIS2 <=> CHAP Core => model services." {
            include *
            autolayout lr
        }

        container chapCore "L2_ChapCore" "Level 2 - Containers inside CHAP Core and how they connect to external systems." {
            include *
            autolayout lr
        }

        container chapkit "L2_Chapkit" "Level 2 - Containers inside a chapkit model service." {
            include *
            autolayout lr
        }

        component chapCore.api "L3_CoreAPI" "Level 3 - CHAP Core REST API components." {
            include *
            autolayout lr
        }

        component chapCore.worker "L3_CoreWorker" "Level 3 - CHAP Core Celery worker components." {
            include *
            autolayout lr
        }

        component chapkit.serviceApi "L3_ChapkitService" "Level 3 - chapkit Service API components." {
            include *
            autolayout lr
        }

        dynamic chapCore "Flow_IngestDataset" "Flow - import and harmonise a dataset from the Modelling App." {
            modellingApp -> chapCore.api "POST /v1/analytics/make-dataset (observations + geojson)"
            chapCore.api -> chapCore.redis "Validate input, then queue harmonise-dataset job"
            chapCore.worker -> chapCore.redis "Fetch job"
            chapCore.worker -> chapCore.db "Harmonise & save DataSet + Observations"
            autolayout lr
        }

        dynamic chapCore "Flow_Backtest" "Flow - run an evaluation (backtest) and read results." {
            modellingApp -> chapCore.api "POST /v1/analytics/create-backtest-with-data"
            chapCore.api -> chapCore.redis "Validate input, then queue backtest job (existing model_id)"
            chapCore.worker -> chapCore.redis "Fetch job"
            chapCore.worker -> chapCore.db "Save DataSet; load ConfiguredModel by id"
            chapCore.worker -> chapkit "Train + predict per fold (chapkit service, or an in-process MLproject runner)"
            chapCore.worker -> chapCore.db "Write Backtest + forecasts + metrics"
            modellingApp -> chapCore.api "GET evaluation results"
            chapCore.api -> chapCore.db "Read forecasts + actuals"
            autolayout lr
        }

        dynamic chapCore "Flow_Prediction" "Flow - run a prediction, review it, and optionally push approved forecasts to DHIS2." {
            modellingApp -> chapCore.api "Request prediction (or a stored PredictionSetup cron fires)"
            chapCore.api -> chapCore.redis "Queue prediction job"
            chapCore.worker -> chapCore.redis "Fetch job"
            chapCore.worker -> chapkit "Predict quantile forecasts (chapkit service, or an in-process MLproject runner)"
            chapCore.worker -> chapCore.db "Store Prediction samples (collected in CHAP)"
            modellingApp -> chapCore.api "Pull prediction quantiles (/v1/analytics/prediction-entry)"
            analyst -> modellingApp "Review forecasts"
            modellingApp -> dhis2 "If approved, write forecast data values (dataValueSets)"
            autolayout lr
        }

        styles {
            element "Person" {
                shape Person
                background #08427b
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
            element "Database" {
                shape Cylinder
                background #438dd5
                color #ffffff
            }
            # Technology logos (C4 leaves icons optional; Structurizr renders them per tag).
            element "PostgreSQL" {
                icon https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg
            }
            element "Redis" {
                icon https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg
            }
            element "SQLite" {
                icon https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg
            }
            element "FastAPI" {
                icon https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg
            }
            element "React" {
                icon https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg
            }
        }
    }
}
