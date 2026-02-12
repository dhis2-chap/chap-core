# API Refactoring: Summary and Follow-up Work

## What we did

### Problem

The backend was served directly from `chap_core.rest_api.v1.rest_api:app` - a single v1 FastAPI app with `root_path="/v1"`. There was no parent app, no versioning structure, and no way to add v2 endpoints alongside v1. All endpoints (health checks, CRUD, analytics, jobs, etc.) were mixed together at the same level.

### Changes

We introduced a proper app structure with a parent app (`chap_core.rest_api.app:app`) that mounts versioned routers:

```
chap_core/rest_api/app.py         <- Single entry point (what Docker/uvicorn serves)
chap_core/common_routes.py        <- Version-independent endpoints (health, status, etc.)
chap_core/rest_api/v1/rest_api.py <- V1-specific routes (crud, analytics, jobs, visualization)
chap_core/rest_api/v2/rest_api.py <- V2 routes (service registration)
```

The parent app includes all routers:

```python
app.include_router(common_router)                                       # /health, /status, /version, ...
app.include_router(v1_router, prefix="/v1")                             # /v1/crud/..., /v1/analytics/..., ...
app.include_router(v2_router, prefix="/v2")                             # /v2/services/...
app.include_router(common_router, prefix="/v1", include_in_schema=False)  # backward compat
```

### API endpoint layout

| Path | Description | Notes |
|---|---|---|
| `/health` | Health check | Common (version-independent) |
| `/status` | Job status | Common |
| `/version` | API version | Common |
| `/is-compatible` | Modelling app compatibility check | Common |
| `/system-info` | System info | Common |
| `/get-results` | Get prediction results | Common |
| `/get-evaluation-results` | Get evaluation results | Common |
| `/get-exception` | Get exception info | Common |
| `/cancel` | Cancel job | Common |
| `/v1/crud/...` | CRUD operations (models, datasets, backtests, predictions) | V1 |
| `/v1/analytics/...` | Analytics (make-dataset, create-backtest, etc.) | V1 |
| `/v1/jobs/...` | Job management | V1 |
| `/v1/visualization/...` | Visualization endpoints | V1 |
| `/v1/debug/...` | Debug endpoints | V1 |
| `/v2/services/...` | Service registration/discovery | V2 (new) |

Common endpoints are also available at `/v1/health`, `/v1/status`, etc. for backward compatibility (hidden from OpenAPI schema).

### Why common endpoints belong at root (`/`)

These endpoints are version-independent. They don't change between API versions:
- `/health` - Is the server running?
- `/status` - What's the current job doing?
- `/version` - What version of chap-core is this?
- `/is-compatible` - Is the frontend compatible?
- `/system-info` - What OS/Python/chap-core version?
- `/get-results`, `/get-evaluation-results`, `/get-exception`, `/cancel` - Job lifecycle

They need to work regardless of which API version the frontend uses. A v2 frontend should still be able to check `/health` and `/status` without knowing about v1.

### Why domain endpoints belong at `/v1/`

CRUD, analytics, jobs, visualization, and debug endpoints are v1-specific implementations. When v2 evolves its own versions of these (different request/response formats, different behavior), they'll go under `/v2/`. Keeping them versioned prevents breaking changes.

### Other changes

- **Dockerfile**: Updated CMD from `chap_core.rest_api.v1.rest_api:app` to `chap_core.rest_api.app:app`
- **Docker Compose**: Added `init: true` to all services for proper signal handling
- **Exception handler**: Single global handler at the parent app level (no duplication)
- **CORS**: Configured once at the parent app level
- **Tests**: All tests updated to use the root app, comprehensive mounting tests added

---

## Follow-up: Frontend needs to use `/v1` prefix in DHIS2 context

**Priority:** High
**Affects:** DHIS2 modeling app integration
**Target:** Next week

### The problem

The old backend had `root_path="/v1"` but was served at `/` (root). This meant `root_path` was cosmetic (only affected OpenAPI docs), and all routes actually lived at `/` on the server. The DHIS2 proxy forwarded requests to CHAP without any `/v1/` prefix, and it worked because routes were at root.

Now that v1 routes are properly at `/v1/`, the DHIS2 frontend gets 404s:

```
GET /api/routes/chap/run/crud/configured-models  -> proxy -> GET /crud/configured-models -> 404
GET /api/routes/chap/run/crud/backtests           -> proxy -> GET /crud/backtests         -> 404
```

These should be:
```
GET /api/routes/chap/run/v1/crud/configured-models  -> proxy -> GET /v1/crud/configured-models -> 200
GET /api/routes/chap/run/v1/crud/backtests           -> proxy -> GET /v1/crud/backtests         -> 200
```

### Root cause

In `apps/modeling-app/src/features/route-api/SetChapUrl.tsx`:

```typescript
OpenAPI.BASE = baseUrl + '/api/routes/chap/run'
```

This replaces the default `BASE: '/v1'` with a proxy URL that has no version prefix. So paths like `/crud/configured-models` are sent without `/v1/`.

### Proposed fix

Update `SetChapUrl.tsx` to append `/v1`:

```typescript
// Before
OpenAPI.BASE = baseUrl + '/api/routes/chap/run'

// After
OpenAPI.BASE = baseUrl + '/api/routes/chap/run/v1'
```

This is the minimal change. Common endpoints (health, status, etc.) will still work at both `/` and `/v1/` thanks to the backward-compat mount.

### Alternative (larger change)

Regenerate the frontend OpenAPI client from the new backend spec and update `BASE` to `/`. This would require mounting v1 routes at root too (or restructuring the API further). Not recommended for now.

---

## Follow-up: OpenAPI spec regeneration

**Priority:** Medium

The frontend's `packages/ui/public/openapi.json` was generated from the old v1-only app. It should be regenerated to include the new API structure.

### Steps

1. Start the backend: `docker compose up`
2. Download: `curl http://localhost:8000/openapi.json > packages/ui/public/openapi.json`
3. Regenerate the TypeScript client
4. Update `OpenAPI.BASE` per the fix above

---

## Follow-up: Deprecated endpoints

**Priority:** Low

These endpoints return empty lists and can be removed once the frontend no longer references them:

- `GET /list-models` - replaced by `GET /v1/crud/model-templates`
- `GET /list-features` - replaced by model template features

Still present in the generated OpenAPI client but may not be actively used by any UI component.
