# Architecture diagrams

This page gives a top-level view of the CHAP system across the three main
repositories — `chap-core` (backend), `chap-frontend` (web UI) and
`automatic-model` (an example chapkit model service) — as a set of mermaid
diagrams.

The diagrams complement the more detailed guides:

- [Code overview](code_overview.md)
- [REST API and database architecture](rest_api_and_database.md)
- [Evaluation pipeline](evaluation_pipeline.md)

## Component diagram

Shows the main components across the three repos and how they relate. The
FastAPI server, the Celery worker, Redis and PostgreSQL all live in
`chap-core`. `chap-frontend` is an embedded DHIS2 app that talks to the REST
API through a generated OpenAPI client. Model execution happens either
in-process via a `TrainPredictRunner` (e.g. Docker, MLflow, UV) or remotely
against a chapkit HTTP service such as `automatic-model`.

```mermaid
flowchart LR
    User([User])

    subgraph DHIS2["DHIS2 instance"]
        DHIS2Core[DHIS2 core]
    end

    subgraph Frontend["chap-frontend (pnpm monorepo)"]
        ModelingApp[modeling-app<br/>React + Vite]
        UIPkg[ui package<br/>OpenAPI client + components]
        ModelingApp --> UIPkg
    end

    subgraph ChapCore["chap-core"]
        subgraph API["FastAPI app (rest_api/app.py)"]
            V1[/v1 routers<br/>crud, analytics, jobs, visualization/]
            V2[/v2 routers<br/>services registry/]
            Common[/common routes<br/>health, info/]
        end

        Worker[Celery worker<br/>db_worker_functions]
        Orchestrator[Orchestrator<br/>rest_api/services]

        subgraph Runners["TrainPredictRunner implementations"]
            Docker[DockerTrainPredictRunner]
            UV[UvTrainPredictRunner]
            Conda[CondaTrainPredictRunner]
            Renv[RenvTrainPredictRunner]
            CLI2[CommandLineTrainPredictRunner]
            MLflow[MlFlowTrainPredictRunner]
        end

        ChapkitClient[CHAPKitRestAPIWrapper<br/>httpx client]
        ChapCLI[chap CLI<br/>cli.py]
    end

    Redis[(Redis<br/>Celery broker +<br/>service registry)]
    Postgres[(PostgreSQL<br/>datasets, backtests,<br/>predictions, models)]
    MLflowDB[(MLflow tracking<br/>mlflow.db)]

    subgraph ExternalModels["External model services (chapkit)"]
        AutoModel[automatic-model<br/>FastAPI + chapkit<br/>RandomForest disease model]
        OtherChapkit[Other chapkit services<br/>e.g. ewars, bayesian]
    end

    ExternalRepos[Model source repos<br/>git / MLproject]

    User --> DHIS2Core
    DHIS2Core --> ModelingApp
    UIPkg -- HTTPS --> API

    V1 -- queue task --> Redis
    V1 -- read/write --> Postgres
    V2 -- register/ping --> Orchestrator
    Orchestrator -- TTL keys --> Redis
    Orchestrator -- sync templates --> Postgres

    Worker -- consume --> Redis
    Worker -- read/write --> Postgres
    Worker -- resolve URL --> Orchestrator
    Worker --> Runners
    Worker --> ChapkitClient

    Runners -- clone / run --> ExternalRepos
    MLflow -- runs --> MLflowDB
    ChapkitClient -- HTTP --> AutoModel
    ChapkitClient -- HTTP --> OtherChapkit
    AutoModel -- $register / $ping --> V2
    OtherChapkit -- $register / $ping --> V2

    ChapCLI --> Runners
```

## Sequence diagram

Representative end-to-end flow for a user-triggered evaluation (backtest)
launched from the modeling app. The same shape applies to predictions
(`/v1/analytics/make-prediction`) — only the worker function and the result
endpoint differ.

```mermaid
sequenceDiagram
    actor User
    participant FE as modeling-app<br/>(chap-frontend)
    participant API as FastAPI<br/>(rest_api/v1)
    participant DB as PostgreSQL
    participant Q as Redis<br/>(Celery broker)
    participant W as Celery worker<br/>(db_worker_functions)
    participant O as Orchestrator<br/>(service registry)
    participant M as Model service<br/>(chapkit / Docker runner)

    User->>FE: configure and submit backtest
    FE->>API: POST /v1/analytics/create-backtest-with-data
    API->>DB: insert DataSet + Observations
    API->>Q: queue run_backtest task
    API-->>FE: 200 JobResponse { id }

    loop poll until terminal
        FE->>API: GET /v1/jobs/{id}
        API->>Q: fetch job_meta:{id}
        API-->>FE: status (PENDING / STARTED / ...)
    end

    Q-->>W: deliver task
    W->>DB: load DataSet + ConfiguredModel
    W->>O: resolve live service URL (if chapkit)
    O-->>W: base_url or stored fallback
    W->>M: train + predict per fold
    M-->>W: samples / forecasts
    W->>DB: insert BackTest + BackTestForecast + BackTestMetric
    W->>Q: set job_meta status = SUCCESS

    FE->>API: GET /v1/jobs/{id}
    API-->>FE: SUCCESS + database_result id
    FE->>API: GET /v1/analytics/evaluation-entry?backtestId=...
    API->>DB: load forecasts + actuals
    API-->>FE: quantile forecast payload
    FE-->>User: render charts and metrics
```

## Class diagram

Main domain classes and their relationships as persisted in PostgreSQL.
All tables inherit from `DBModel` (SQLModel + camelCase aliasing). `BackTest`
and `Prediction` are the two "run" concepts — a `BackTest` is a retrospective
evaluation over known data, a `Prediction` is a forward forecast. Both carry
forecast samples and reference the `DataSet` and `ConfiguredModelDB` they were
produced from.

```mermaid
classDiagram
    class DBModel {
        <<base, SQLModel>>
        +id: int
    }

    class DataSet {
        +name: str
        +type: str
        +geojson: str
        +created: datetime
    }

    class Observation {
        +period: str
        +orgUnit: str
        +element: str
        +value: float
    }

    class ModelTemplateDB {
        +name: str
        +sourceUrl: str
        +description: str
    }

    class ConfiguredModelDB {
        +name: str
        +userOptions: dict
        +additionalContinuousCovariates: list
        +deleted: bool
    }

    class BackTest {
        +name: str
        +created: datetime
        +aggregationLevel: str
        +splitPeriods: list
    }

    class BackTestForecast {
        +period: str
        +orgUnit: str
        +lastTrainPeriod: str
        +values: list~float~
    }

    class BackTestMetric {
        +metricId: str
        +period: str
        +orgUnit: str
        +value: float
    }

    class Prediction {
        +name: str
        +created: datetime
        +metaData: dict
    }

    class PredictionSamplesEntry {
        +period: str
        +orgUnit: str
        +values: list~float~
    }

    class DebugEntry {
        +timestamp: datetime
        +payload: dict
    }

    DBModel <|-- DataSet
    DBModel <|-- Observation
    DBModel <|-- ModelTemplateDB
    DBModel <|-- ConfiguredModelDB
    DBModel <|-- BackTest
    DBModel <|-- BackTestForecast
    DBModel <|-- BackTestMetric
    DBModel <|-- Prediction
    DBModel <|-- PredictionSamplesEntry
    DBModel <|-- DebugEntry

    DataSet "1" --> "*" Observation : observations
    ModelTemplateDB "1" --> "*" ConfiguredModelDB : configurations
    DataSet "1" --> "*" BackTest : backtests
    DataSet "1" --> "*" Prediction : predictions
    ConfiguredModelDB "1" --> "*" BackTest : runs
    ConfiguredModelDB "1" --> "*" Prediction : runs
    BackTest "1" --> "*" BackTestForecast : forecasts
    BackTest "1" --> "*" BackTestMetric : metrics
    Prediction "1" --> "*" PredictionSamplesEntry : samples
```
