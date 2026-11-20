# Auth contract

This page is the contract chap-core's authentication and authorization are built and tested
against (epic CLIM-1955). It is design: until the tasks that implement it ship, chap-core
behaves as described in [API Authentication](../webapi/api-authentication.md).

chap-core takes Open Geo Service's contract
([`docs/authentication.md`](https://github.com/dhis2/open-geo-service/blob/main/docs/authentication.md),
CLIM-1292) and adds permissions. The two share the format of the `authentication` block, the
headers and the error answers. Each service has its own tiers: Open Geo Service decides by
route tier, Chap by permission. An Open Geo Service block is valid in `chap-server.yaml`,
because the tiers it can give a token, `authenticated` and `operator`, exist in Chap too.

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
| A chapkit service registering itself | `SERVICEKIT_REGISTRATION_KEY` | `service-registry` |

Usernames and email addresses are never principal ids. Records store the principal id
(CLIM-1965).

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
`chap-server.yaml` with its secret in the environment:

```yaml
authentication:
  tokens:
    - id: modeling-app-read
      token_env: CHAP_TOKEN_MODELING_APP_READ
      tier: read
    - id: modeling-app
      token_env: CHAP_TOKEN_MODELING_APP
      tier: authenticated
    - id: modeling-app-admin
      token_env: CHAP_TOKEN_MODELING_APP_ADMIN
      tier: operator
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
- a token names an unknown tier.

The startup error names the token by position and id, as `authentication.tokens[1]
(modeling-app)`, and never prints a secret.

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

- The set is fixed in code (CLIM-1959). A deployment cannot add one.
- Actions are `read`, `write`, `delete`, `run`, `approve`, `call` and `register`. A new
  action needs a change to this contract.
- Each route needs exactly one permission, chosen per route rather than per HTTP method. A
  `POST` that only reads, such as a query with a body, needs a `read:` permission.
- Routes declare it with FastAPI's `Security(authorize, scopes=["read:predictions"])`, so it
  shows in `/docs` and in the OpenAPI spec.
- chap-core refuses to start unless every operation in the OpenAPI spec is public or
  declares exactly one permission. Routes left out of the spec come from a short explicit
  list: the chapkit proxy (`/v2/services/{service_id}/run/{path}`, `call:services`), the
  docs pages and `/favicon.ico`.

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
| GET | `/v1/crud/datasets/{datasetId}` | `read:datasets` |
| DELETE | `/v1/crud/datasets/{datasetId}` | `delete:datasets` |
| POST | `/v1/crud/datasets/csvFile` | `write:datasets` |
| GET | `/v1/crud/datasets/{datasetId}/df` | `read:datasets` |
| GET | `/v1/crud/datasets/{datasetId}/csv` | `read:datasets` |
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
| POST | `/v2/services/$register` | `register:services` |
| PUT | `/v2/services/{service_id}/$ping` | `register:services` |
| GET | `/v2/services` | `read:services` |
| GET | `/v2/services/{service_id}` | `read:services` |
| DELETE | `/v2/services/{service_id}` | `register:services` |
| GET | `/v2/services/{service_id}/run/{path}` | `call:services` |
| GET | `/favicon.ico` | public |

## Built-in tiers

| Tier | Permissions |
| --- | --- |
| `public` | Only the open routes. Given to `anonymous`; cannot be given to a token. |
| `read` | Every `read:` permission except `read:jobs`, because job logs and requests can hold raw data, and `read:system`. For dashboards and the read-only DHIS2 route. |
| `authenticated` | `read`, plus `read:jobs`, `write:jobs`, `read:system`, `write:datasets`, `run:backtests`, `run:predictions`, `write:prediction-setups` and `call:services`. Not deleting, and not changing models, settings, backtests or alerts. |
| `operator` | Every permission, including ones added in later versions. |

The catalogue (CLIM-1959) lists each tier's permissions exactly. The names follow Open Geo
Service where the idea is the same, but what a tier grants is Chap's own.

### The public tier

These routes are open with authentication on:

- `/health` and `/health/ready`, which container health checks call without headers,
- `/system/info`, which tells a client whether a token is required (`auth_required`),
- `/docs`, `/docs/oauth2-redirect`, `/redoc` and `/openapi.json`, the API reference,
- `/favicon.ico`.

They are listed in one set in code (`PUBLIC_PATHS`), which the startup check reads, so a
route is public only on purpose. `/health/probe` stays gated, as today: it round-trips the
broker and the database, so it needs `read:system`.

## Custom tiers

Chap's addition. An optional `tiers` block under `authentication` defines tiers by name:

```yaml
authentication:
  tiers:
    analyst:
      includes: [read]
      permissions: [run:backtests, run:predictions]
    model-curator:
      includes: [analyst]
      permissions: [write:models]
  tokens:
    - id: notebook
      token_env: CHAP_TOKEN_NOTEBOOK
      tier: analyst
```

- `includes` lists tiers whose permissions this one gets. `permissions` lists more.
- Both are optional, but a tier needs at least one of them.
- A tier name matches the token id pattern.

chap-core refuses to start when the block has:

- an unknown permission or an unknown included tier,
- a tier named `public`, `read`, `authenticated` or `operator`,
- a cycle in `includes`,
- a wildcard, such as `read:*` or `*`,
- an unknown key.

Tiers are resolved once at startup into a flat set of permissions per tier.

## Existing installs

These keep today's behaviour. Nothing has to change on upgrade.

- **No auth configured:** everything is open.
- **`SERVICEKIT_REGISTRATION_KEY` only:** registering, pinging and deregistering a service
  (`POST /v2/services/$register`, `PUT /v2/services/{id}/$ping`,
  `DELETE /v2/services/{id}`) need the key in `X-Service-Key`. Everything else is open.
- **`CHAP_API_TOKEN`:** read as an operator token with the id `legacy`. It is accepted in
  every header in [Sending a credential](#sending-a-credential), including `X-Service-Key` on
  any route. A secret shorter than 32 characters still only logs a warning, as today; the
  32-character rule applies to named tokens.
- **Both `CHAP_API_TOKEN` and `SERVICEKIT_REGISTRATION_KEY`:** the three registration routes
  need the registration key in `X-Service-Key`. An operator token alone is refused there, as
  today.

The `CHAP_API_TOKEN` token can sit beside named tokens, so an install can move clients to
named tokens one at a time.

One visible change: a 401 body gains the `id`, `code` and `message` fields, and its `detail`
text changes. The status codes and the `WWW-Authenticate` header stay the same.

## Where it is enforced

- Starlette's `AuthenticationMiddleware` with a `ChapTokenBackend` finds the credential and
  resolves the principal (CLIM-1961). It answers 401 before the body is read.
- `Security(authorize, scopes=[...])` on each route compares the principal's permissions with
  the route's and answers 403 (CLIM-1962).
- The registration key check stays a dependency on the three registration routes.
- `GET /v1/auth/me` reports the caller's principal, tier and permissions, and needs no
  permission (CLIM-1964).
- The conformance tests (CLIM-1960) check every row of this page against the running app.

## Decided here

- **Basic `id:secret`:** accepted, as in Open Geo Service. It costs one branch, keeps both
  services answering the same clients, and the id must match the secret's token.
- **Deletes:** every `delete:` permission is in `operator` only. A custom tier can grant
  one. Deleting your own records waits until records store their principal (CLIM-1965);
  records from before that have no owner.

## Proposal to Open Geo Service

To keep one format for both services, Open Geo Service would accept the `tiers` block in the
same shape, with its own tier and permission names, if it adds custom tiers. Until then Open
Geo Service rejects the block as an unknown key.
