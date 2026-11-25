# Auth contract

This page is the contract chap-core's authentication and authorization are built and tested
against. [API Authentication](../webapi/api-authentication.md) covers the
single `CHAP_API_TOKEN` from the point of view of whoever runs Chap.

chap-core takes Open Geo Service's contract
([`docs/authentication.md`](https://github.com/dhis2/open-geo-service/blob/main/docs/authentication.md))
and adds permissions. The two share the format of the `authentication` block, the
headers and the error answers. Each service has its own tiers: Open Geo Service decides by
route tier, Chap by permission. The tier names are Chap's own, so a token copied from an Open
Geo Service block needs a Chap tier.

## Principals, tiers and permissions

A **principal** says who is calling. A **tier** says what that principal may do. A
**permission** is one thing a route needs, such as `read:predictions`. A tier is a set of
permissions, and every principal has exactly one tier.

| Caller | Credential | Principal id |
| --- | --- | --- |
| Anyone, with authentication on | none | `anonymous` |
| Anyone, with authentication off | none | `open` |
| Machine client, the Modeling App's DHIS2 routes, scripts | a named token | `token:<id>` |
| Existing installs | `CHAP_API_TOKEN` | `token:legacy` |
| A chapkit service registering itself | `SERVICEKIT_REGISTRATION_KEY` in `X-Service-Key`, on `/v2/services` only | `service-registry`, with `register:services`, `read:services` and `call:services`, the registry routes it opened before named tokens |

Usernames and email addresses are never principal ids. Records store the principal id.

## Turning authentication on

Authentication is on when `CHAP_API_TOKEN` is set or `authentication.tokens` lists at least
one token, and off otherwise.

- **Off:** no credential is checked and no permission is enforced. Every request runs as
  `open` with every permission. This is today's default, right for a laptop and for a
  Chap that only DHIS2 can reach.
- **On:** a request without a credential runs as `anonymous` with the `public` tier.

`SERVICEKIT_REGISTRATION_KEY` alone does not turn authentication on. It gates the service
registry routes only, as today (see [Existing installs](#existing-installs)).

## Named tokens

Taken unchanged from Open Geo Service. Each client gets its own token, listed in
`chap-server.yml` with its secret in the environment:

```yaml
authentication:
  tokens:
    - id: modeling-app-read
      token_env: CHAP_TOKEN_MODELING_APP_READ
      tier: viewer
    - id: modeling-app
      token_env: CHAP_TOKEN_MODELING_APP
      tier: analyst
    - id: modeling-app-admin
      token_env: CHAP_TOKEN_MODELING_APP_ADMIN
      tier: admin
```

- `id` matches `^[A-Za-z0-9][A-Za-z0-9_-]*$`. It is stable, not secret, and becomes the
  principal `token:<id>`.
- `token_env` names the environment variable holding the secret, and matches
  `^[A-Za-z_][A-Za-z0-9_]*$`. A `.env` file works, because chap-core copies it into the
  environment. The block takes no `${...}` interpolation.
- `tier` is one tier name. A token cannot be given permissions directly, and `public` cannot
  be given to a token.
- Unknown keys are rejected.

chap-core refuses to start when:

- a token's variable is unset or empty,
- a secret is shorter than 32 characters (`openssl rand -hex 32` makes a suitable one),
- two tokens share an id or a secret, including the `CHAP_API_TOKEN` secret,
- a token has the id `legacy`, which `CHAP_API_TOKEN` uses,
- a token names an unknown tier,
- `SERVICEKIT_REGISTRATION_KEY` is also the secret of a named token. It may equal
  `CHAP_API_TOKEN`, as existing installs set it.

Each startup error names what is wrong: a token by position and id, as
`authentication.tokens[1] (modeling-app)`, or two tokens by their ids when they repeat. It
never prints a secret. A `token_env` that is not an upper-case variable name, or a tier name
as long as a secret, is left out of the message, in case a secret was pasted there.
Surrounding whitespace in a secret is ignored, so a secret of only whitespace counts as unset.

Secrets are compared in constant time against every token, so the time taken does not show
which one came close. Secrets are never logged. Rejected credentials are logged at most once
a minute, with a count.

## Sending a credential

| Header | Accepted for | Source |
| --- | --- | --- |
| `Authorization: Bearer <secret>` | every token | Open Geo Service |
| `Authorization: Bearer basic//<secret>` | every token | Open Geo Service (openEO clients) |
| `Authorization: Basic <base64 id:secret>` | every token; the id must match the secret's token | Open Geo Service |
| `Authorization: ApiToken <secret>` | every token | Chap today (DHIS2 route `api-token` auth) |
| `X-Service-Key: <secret>` | every token, and the registration key on `/v2/services` | Chap today (servicekit cannot send `Authorization`) |

Scheme names are case-insensitive. When both headers are sent, a valid `Authorization`
decides the principal; otherwise a valid `X-Service-Key` does, as in Chap today.

## Answers

Taken from Open Geo Service. Every refusal has the same body, which keeps FastAPI's `detail`
and adds the openEO error fields:

```json
{"id": "authentication", "code": "AuthenticationRequired", "message": "...", "detail": "..."}
```

| Case | Status | `code` |
| --- | --- | --- |
| A credential that matches no token | 401 | `TokenInvalid` |
| No credential, on a route `anonymous` may not call | 401 | `AuthenticationRequired` |
| A valid token without the route's permission | 403 | `PermissionsInsufficient` |

A 401 carries `WWW-Authenticate: Bearer`. The message names the route and what is missing:
`POST /v1/analytics/create-backtest needs the run:backtests permission, which
token:modeling-app-read does not have.`

The middleware sits inside CORS, so a refusal still carries the headers a browser needs, and
`OPTIONS` preflights pass. A 401 is returned before the request body is read.

One difference from Open Geo Service: the open routes (see [The public tier](#the-public-tier))
never refuse. An invalid credential there is ignored, as in Chap today, so a health check
behind a proxy that adds a stale header keeps working.

## Permissions

A permission is named `action:resource`, such as `read:predictions` or `run:backtests`.

- The set is fixed in code. A deployment cannot add one.
- Actions are `read`, `export`, `write`, `delete`, `run`, `approve`, `call` and `register`.
  A new action needs a change to this contract.
- Each route needs exactly one permission, chosen per route rather than per HTTP method. A
  `POST` that only reads, such as a query with a body, needs a `read:` permission.
- A route whose request can also do a second thing checks a second permission in the
  handler, and answers 403 naming it:
    - attaching or changing a prediction setup's alert policy also needs `write:alerts`;
    - running a prediction setup (`POST /v1/crud/prediction-setups/{id}/run`) and the
      deprecated `make-prediction` and `create-backtest-with-data` store the posted
      observations as a dataset, so they also need `write:datasets` (not for a dry run).
    - downloading a job's stored request (`GET /v1/jobs/{job_id}/request`) returns the
      observations it posted, so it also needs `export:datasets`.
- Cancelling or forgetting a job (`write:jobs`) works only on the caller's own jobs, unless
  the caller holds every permission. For another principal's job, or one started while
  authentication was off, cancel and forget answer 404. Reading jobs, their logs and stored
  requests (`read:jobs`, plus `export:datasets` for a stored request) is not limited to the
  caller's own yet.
- Routes declare it with FastAPI's `Security(authorize, scopes=["read:predictions"])`, so it
  shows in `/docs` and in the OpenAPI spec.
- chap-core refuses to start unless every route it serves is public or declares exactly one
  permission. The check walks the routes themselves, so routes left out of the OpenAPI spec,
  such as the chapkit proxy, are covered too.

Every permission, what it allows, and which built-in tiers hold it:

<div class="permission-table" markdown>

| Permission | Allows | `viewer` | `analyst` | `admin` |
| --- | --- | --- | --- | --- |
| `read:datasets` | Listing datasets, dataset plots, computing thresholds | yes | yes | yes |
| `export:datasets` | A dataset's observations: the dataset itself, and its `/df` and `/csv` downloads | | yes | yes |
| `write:datasets` | Importing datasets | | yes | yes |
| `delete:datasets` | Deleting datasets | | | yes |
| `read:models` | Listing model templates and configured models | yes | yes | yes |
| `write:models` | Adding model templates and configuring models | | | yes |
| `delete:models` | Retiring model templates and configured models | | | yes |
| `read:backtests` | Backtests, their forecasts, metrics and plots | yes | yes | yes |
| `run:backtests` | Starting backtests | | yes | yes |
| `write:backtests` | Renaming and editing backtests | | | yes |
| `delete:backtests` | Deleting backtests | | | yes |
| `read:predictions` | Predictions and their forecasts | yes | yes | yes |
| `run:predictions` | Running predictions, and prediction setups | | yes | yes |
| `delete:predictions` | Deleting predictions | | | yes |
| `read:prediction-setups` | Prediction setups and their monitoring | yes | yes | yes |
| `write:prediction-setups` | Creating and changing prediction setups; their alert policy also needs `write:alerts` | | yes | yes |
| `delete:prediction-setups` | Retiring prediction setups | | | yes |
| `read:alerts` | Alerts and alert policies | yes | yes | yes |
| `write:alerts` | Recording alerts, creating and changing alert policies | | | yes |
| `approve:alerts` | Clearing or declining alerts for release | | | yes |
| `delete:alerts` | Deleting alerts and alert policies | | | yes |
| `read:jobs` | Jobs, their logs, requests and results | | yes | yes |
| `write:jobs` | Cancelling and forgetting the caller's own jobs; any job with every permission | | yes | yes |
| `read:settings` | The installation's defaults, and the fixed lists clients choose from: plot types, weather providers, data sources, threshold strategies, outbreak metrics, terminology | yes | yes | yes |
| `write:settings` | Changing the installation's defaults | | | yes |
| `read:system` | The on-demand probe, `/health/probe` | | yes | yes |
| `read:services` | Listing registered chapkit services | yes | yes | yes |
| `call:services` | The read-only proxy to a registered chapkit service | | yes | yes |
| `register:services` | Registering, pinging and deregistering a chapkit service | | | yes |

</div>

## Permission catalogue

Every route and the permission it needs, generated from the app. `read:settings` also covers
the fixed lists clients choose from, such as plot types, weather providers and terminology.

| Method | Path | Permission |
| --- | --- | --- |
| GET | `/health` | public |
| GET | `/health/ready` | public |
| GET | `/health/probe` | `read:system` |
| GET | `/system/info` | public |
| GET | `/v1/crud/backtests` | `read:backtests` |
| DELETE | `/v1/crud/backtests` | `delete:backtests` |
| GET | `/v1/crud/backtest-specifications` | `read:backtests` |
| GET | `/v1/crud/backtest-specifications/{specificationId}` | `read:backtests` |
| GET | `/v1/crud/backtest-specifications/{specificationId}/selection-bias` | `read:backtests` |
| GET | `/v1/crud/backtests/{backtestId}/full` | `read:backtests` |
| GET | `/v1/crud/backtests/{backtestId}` | `read:backtests` |
| DELETE | `/v1/crud/backtests/{backtestId}` | `delete:backtests` |
| PATCH | `/v1/crud/backtests/{backtestId}` | `write:backtests` |
| GET | `/v1/crud/backtests/{backtestId}/info` | `read:backtests` |
| GET | `/v1/crud/metric/csv` | `read:backtests` |
| GET | `/v1/crud/predictions` | `read:predictions` |
| GET | `/v1/crud/predictions/{predictionId}` | `read:predictions` |
| DELETE | `/v1/crud/predictions/{predictionId}` | `delete:predictions` |
| GET | `/v1/crud/datasets` | `read:datasets` |
| POST | `/v1/crud/datasets` | `write:datasets` |
| GET | `/v1/crud/datasets/{datasetId}` | `export:datasets` |
| DELETE | `/v1/crud/datasets/{datasetId}` | `delete:datasets` |
| POST | `/v1/crud/datasets/csvFile` | `write:datasets` |
| GET | `/v1/crud/datasets/{datasetId}/df` | `export:datasets` |
| GET | `/v1/crud/datasets/{datasetId}/csv` | `export:datasets` |
| GET | `/v1/crud/model-templates` | `read:models` |
| POST | `/v1/crud/model-templates/from-service` | `write:models` |
| DELETE | `/v1/crud/model-templates/{modelTemplateId}` | `delete:models` |
| GET | `/v1/crud/configured-models` | `read:models` |
| POST | `/v1/crud/configured-models` | `write:models` |
| GET | `/v1/crud/configured-models/{configuredModelId}` | `read:models` |
| DELETE | `/v1/crud/configured-models/{configuredModelId}` | `delete:models` |
| POST | `/v1/crud/alerts` | `write:alerts` |
| GET | `/v1/crud/alerts` | `read:alerts` |
| GET | `/v1/crud/alerts/{alertId}` | `read:alerts` |
| DELETE | `/v1/crud/alerts/{alertId}` | `delete:alerts` |
| POST | `/v1/crud/alerts/$approve` | `approve:alerts` |
| GET | `/v1/crud/alert-policies` | `read:alerts` |
| POST | `/v1/crud/alert-policies` | `write:alerts` |
| GET | `/v1/crud/alert-policies/{alertPolicyId}` | `read:alerts` |
| PATCH | `/v1/crud/alert-policies/{alertPolicyId}` | `write:alerts` |
| DELETE | `/v1/crud/alert-policies/{alertPolicyId}` | `delete:alerts` |
| GET | `/v1/crud/prediction-setups` | `read:prediction-setups` |
| POST | `/v1/crud/prediction-setups` | `write:prediction-setups` |
| GET | `/v1/crud/prediction-setups/{predictionSetupId}` | `read:prediction-setups` |
| PATCH | `/v1/crud/prediction-setups/{predictionSetupId}` | `write:prediction-setups` |
| DELETE | `/v1/crud/prediction-setups/{predictionSetupId}` | `delete:prediction-setups` |
| GET | `/v1/crud/prediction-setups/{predictionSetupId}/monitoring` | `read:prediction-setups` |
| POST | `/v1/crud/prediction-setups/{predictionSetupId}/run` | `run:predictions` |
| POST | `/v1/analytics/make-dataset` | `write:datasets` |
| GET | `/v1/analytics/compatible-backtests/{backtestId}` | `read:backtests` |
| GET | `/v1/analytics/backtest-overlap/{backtestId1}/{backtestId2}` | `read:backtests` |
| GET | `/v1/analytics/prediction-entry` | `read:predictions` |
| GET | `/v1/analytics/evaluation-entry` | `read:backtests` |
| POST | `/v1/analytics/create-backtest` | `run:backtests` |
| POST | `/v1/analytics/create-backtests` | `run:backtests` |
| POST | `/v1/analytics/make-prediction` | `run:predictions` |
| GET | `/v1/analytics/prediction-entry/{predictionId}` | `read:predictions` |
| GET | `/v1/analytics/actualCases/{backtestId}` | `read:backtests` |
| GET | `/v1/analytics/actual-cases/{backtestId}` | `read:backtests` |
| GET | `/v1/analytics/data-sources` | `read:settings` |
| GET | `/v1/analytics/covariate-names` | `read:settings` |
| POST | `/v1/analytics/create-backtest-with-data/` | `run:backtests` |
| GET | `/v1/analytics/backtest-parameters` | `read:settings` |
| GET | `/v1/analytics/weather-providers` | `read:settings` |
| GET | `/v1/analytics/thresholds/strategies` | `read:settings` |
| POST | `/v1/analytics/thresholds` | `read:datasets` |
| GET | `/v1/analytics/outbreak-metrics` | `read:settings` |
| GET | `/v1/analytics/backtests/{backtestId}/outbreak-metrics` | `read:backtests` |
| GET | `/v1/jobs/{job_id}/diagnostics` | `read:jobs` |
| GET | `/v1/jobs` | `read:jobs` |
| GET | `/v1/jobs/{job_id}` | `read:jobs` |
| DELETE | `/v1/jobs/{job_id}` | `write:jobs` |
| GET | `/v1/jobs/{job_id}/request` | `read:jobs` |
| POST | `/v1/jobs/{job_id}/cancel` | `write:jobs` |
| GET | `/v1/jobs/{job_id}/logs` | `read:jobs` |
| GET | `/v1/jobs/{job_id}/prediction_result` | `read:jobs` |
| GET | `/v1/jobs/{job_id}/evaluation_result` | `read:jobs` |
| GET | `/v1/jobs/{job_id}/database_result` | `read:jobs` |
| GET | `/v1/visualization/metric-plots/{backtest_id}` | `read:backtests` |
| GET | `/v1/visualization/metrics/{backtest_id}` | `read:backtests` |
| GET | `/v1/visualization/metric-plots/{visualization_name}/{backtest_id}/{metric_id}` | `read:backtests` |
| GET | `/v1/visualization/dataset-plots/` | `read:settings` |
| GET | `/v1/visualization/dataset-plots/{visualization_name}/{dataset_id}` | `read:datasets` |
| GET | `/v1/visualization/backtest-plots/` | `read:settings` |
| GET | `/v1/visualization/backtest-plots/{visualization_name}/{backtest_id}` | `read:backtests` |
| GET | `/v1/visualization/backtest-plots/{visualization_name}/{backtest_id}/facet-coords` | `read:backtests` |
| POST | `/v1/visualization/backtest-plots/{visualization_name}/{backtest_id}/subplot` | `read:backtests` |
| GET | `/v1/visualization/backtest-plots/{visualization_name}/{backtest_id}/subplots` | `read:backtests` |
| GET | `/v1/settings` | `read:settings` |
| PUT | `/v1/settings` | `write:settings` |
| GET | `/v1/terminology` | `read:settings` |
| GET | `/v1/terminology/{name}` | `read:settings` |
| GET | `/v1/auth/me` | public |
| POST | `/v2/services/$register` | `register:services` |
| PUT | `/v2/services/{service_id}/$ping` | `register:services` |
| GET | `/v2/services` | `read:services` |
| GET | `/v2/services/{service_id}` | `read:services` |
| DELETE | `/v2/services/{service_id}` | `register:services` |
| GET | `/v2/services/{service_id}/run/{path:path}` | `call:services` |
| GET | `/favicon.ico` | public |

## Fixed in code and configured per install

Fixed in code, the same on every install, and changed only by a Chap release:

- the permissions, and which permission each route needs,
- the built-in tiers `public`, `viewer`, `analyst` and `admin`, and what they grant,
- the public routes,
- the accepted headers and the error answers.

Configured per install, in the `authentication` block of `chap-server.yml` (or, like any
setting, as JSON in the `CHAP_AUTHENTICATION` environment variable, which replaces the block):

- the named tokens, each with one tier,
- custom tiers, built from the permissions and the built-in tiers.

A tier is resolved against the running release, and the built-in tiers can widen:

- `admin` gains every permission a later release adds.
- `viewer` gains every new `read:` permission, except ones a release keeps out of it as it
  keeps `read:jobs`; `analyst` includes `viewer`.
- A custom tier that includes a built-in tier widens with it. Only permissions listed one by
  one stay exactly as written, so check custom tiers after upgrading.

## Built-in tiers

| Tier | Permissions |
| --- | --- |
| `public` | Only the open routes. Given to `anonymous`; cannot be given to a token. |
| `viewer` | Every `read:` permission except `read:jobs` and `read:system`. It reads stored results: backtests, their forecasts and the observed cases they were scored against, predictions, models, alerts and settings. It does not hold `export:datasets`, so it cannot download a dataset's observations. Jobs are left out because their logs and stored requests show other clients' work and internal details such as file paths and errors. For dashboards and the read-only DHIS2 route. |
| `analyst` | `viewer`, plus `export:datasets`, `read:jobs`, `write:jobs`, `read:system`, `write:datasets`, `run:backtests`, `run:predictions`, `write:prediction-setups` and `call:services`. It deletes no stored record and changes no models, settings, backtests, alerts or alert policies, and it cancels and forgets only its own jobs. |
| `admin` | Every permission, including ones added in later versions. |

The catalogue lists each tier's permissions exactly. The names say who a token
is for: a `viewer` reads results, an `analyst` runs work, an `admin` may do everything. They
match the DHIS2 routes the Modeling App will use, one per audience (`chap-read`, `chap` and
`chap-admin`). Open Geo Service names its tiers by what a caller has shown (`authenticated`),
so its names are not used here.

### The public tier

These routes are open with authentication on:

- `/health` and `/health/ready`, which container health checks call without headers,
- `/system/info`, which tells a client whether a token is required (`auth_required`),
- `/docs`, `/docs/oauth2-redirect`, `/redoc` and `/openapi.json`, the API reference,
- `/favicon.ico`,
- `/v1/auth/me`, which reports `anonymous` to a caller without a token.

They are listed in one set in code (`PUBLIC_PATHS`), which the startup check reads, so a
route is public only on purpose. `/health/probe` stays gated, as today: it round-trips the
broker and the database, so it needs `read:system`.

## Custom tiers

Chap's addition. An optional `tiers` block under `authentication` defines tiers by name:

```yaml
authentication:
  tiers:
    forecaster:
      includes: [viewer]
      permissions: [run:backtests, run:predictions]
    model-curator:
      includes: [forecaster]
      permissions: [write:models]
  tokens:
    - id: notebook
      token_env: CHAP_TOKEN_NOTEBOOK
      tier: forecaster
```

- `includes` lists tiers whose permissions this one gets. `permissions` lists more.
- Both are optional, but a tier needs at least one of them.
- A tier name matches the token id pattern.

chap-core refuses to start when the block has:

- an unknown permission or an unknown included tier,
- a tier named `public`, `viewer`, `analyst` or `admin`,
- a cycle in `includes`,
- a wildcard, such as `read:*` or `*`,
- an unknown key.

Tiers are resolved once at startup into a flat set of permissions per tier.

Names never replace each other:

- A custom tier cannot take a built-in name. A tier named `viewer` stops startup rather than
  replacing the built-in one.
- A tier name, or any other key, given twice in `chap-server.yml` stops startup and names
  the line. Plain YAML would keep the last one silently.
- A token id given twice stops startup and names the id.
- Tier names and token ids are separate, so a token and a tier may share a name, as `caps`
  does in the example below.

### Making a tier

1. Start from the clients and what each must do, not from the permissions. Give every client
   its own token, even when two need the same access.
2. Use a built-in tier when one fits. Make a custom tier only for a client that needs less
   than `analyst` or `admin` gives, or something in between.
3. Build it from `includes: [viewer]` and add the few permissions the client needs from
   [Permissions](#permissions). `viewer` and every other built-in tier widen when a release adds
   permissions, so for a client that must never gain access on upgrade, list every permission
   it needs and include no tier.
4. Leave out `delete:` permissions unless the client must delete. They are admin-only by
   default.
5. Start Chap and call `GET /v1/auth/me` with the token. It lists exactly what the token may
   do. An answer of `anonymous` means the token did not match: `/v1/auth/me` never refuses.

For example, a forecasting client that imports data and runs predictions, but must not
change models or delete anything:

```yaml
authentication:
  tiers:
    caps:
      includes: [viewer]
      permissions: [write:datasets, run:predictions, read:jobs]
  tokens:
    - id: caps
      token_env: CHAP_CAPS_TOKEN
      tier: caps
```

## Existing installs

These keep today's behaviour. Nothing has to change on upgrade.

- **No auth configured:** everything is open.
- **`SERVICEKIT_REGISTRATION_KEY` only:** registering, pinging and deregistering a service
  (`POST /v2/services/$register`, `PUT /v2/services/{id}/$ping`,
  `DELETE /v2/services/{id}`) need the key in `X-Service-Key`. Everything else is open.
- **`CHAP_API_TOKEN`:** read as an admin token with the id `legacy`. It is accepted in
  every header in [Sending a credential](#sending-a-credential), including `X-Service-Key` on
  any route. A secret shorter than 32 characters still only logs a warning, as today; the
  32-character rule applies to named tokens.
- **Both `CHAP_API_TOKEN` and `SERVICEKIT_REGISTRATION_KEY`:** the three registration routes
  need the registration key in `X-Service-Key`. An admin token alone is refused there, as
  today.

The `CHAP_API_TOKEN` token can sit beside named tokens, so an install can move clients to
named tokens one at a time.

One visible change: a 401 body gains the `id`, `code` and `message` fields, and its `detail`
text changes. The status codes and the `WWW-Authenticate` header stay the same.

## Where it is enforced

- Starlette's `AuthenticationMiddleware` with a `ChapTokenBackend` finds the credential and
  resolves the principal. It answers 401 before the body is read.
- `Security(authorize, scopes=[...])` on each route compares the principal's permissions with
  the route's and answers 403.
- The registration key check stays a dependency on the three registration routes.
- `GET /v1/auth/me` reports the caller's principal, tier and permissions, and needs no
  permission.
- The conformance tests check the rules on this page against the running app,
  every route refuses a token without its permission, and a test compares the permission and
  route tables here with the code.

## Decided here

- **Basic `id:secret`:** accepted, as in Open Geo Service. It costs one branch, keeps both
  services answering the same clients, and the id must match the secret's token.
- **Deletes:** every `delete:` permission is in `admin` only. A custom tier can grant
  one. Deleting your own records waits until records store their principal;
  records from before that have no owner.

## Proposal to Open Geo Service

To keep one format for both services, Open Geo Service would accept the `tiers` block in the
same shape, with its own tier and permission names, if it adds custom tiers. Until then Open
Geo Service rejects the block as an unknown key.
