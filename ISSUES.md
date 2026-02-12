# Known Issues

## Frontend OpenAPI base URL still uses `/v1`

The frontend (`chap-frontend`) has `OpenAPI.BASE = '/v1'` hardcoded in `packages/ui/src/httpfunctions/core/OpenAPI.ts`. This means all API calls get `/v1/` prepended.

With the new app structure, common endpoints (health, status, version, is-compatible, etc.) live at root level, while v1-specific endpoints (crud, analytics, jobs, visualization) remain under `/v1/`.

**Action needed:** Update the frontend's OpenAPI base URL from `/v1` to `/` (or empty string), and regenerate the OpenAPI client. The frontend's `openapi.json` spec at `packages/ui/public/openapi.json` should also be regenerated from the new backend.

Paths that moved from `/v1/` to root:
- `/status`
- `/health`
- `/version`
- `/is-compatible`
- `/system-info`
- `/get-results`
- `/get-evaluation-results`
- `/get-exception`
- `/cancel`
- `/list-models` (deprecated)
- `/list-features` (deprecated)

Paths that stay under `/v1/`:
- `/v1/crud/...`
- `/v1/analytics/...`
- `/v1/jobs/...`
- `/v1/visualization/...`
- `/v1/debug/...`

**Temporary workaround:** Common routes are also registered under `/v1/` prefix with `include_in_schema=False` for backward compatibility until the frontend is updated.
