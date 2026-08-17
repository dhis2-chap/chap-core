# API Authentication

By default the Chap Core API has **no authentication**. This is intentional: the supported
deployments either run DHIS2 and Chap Core together in one containerised setup, or run both
locally during development. In both cases the API is reachable only from the machine or the
internal network, and DHIS2 enforces authorization in front of it via the Route API.

If you need to expose Chap Core beyond that -- for example connecting a cloud DHIS2 instance
to a local Chap Core through a tunnel -- you can enable an opt-in API token. It is the bare
minimum needed to put the API somewhere it can be reached from the internet, and it should
always be combined with HTTPS.

!!! warning "A token is not a substitute for network isolation"
    The token is a single shared secret with no rotation, no expiry and no per-client
    identity. Keeping Chap Core off the public internet is still the recommended setup.
    See [Recommendation for server deployment](../modeling-app/running-chap-on-server.md).

## Enabling the token

Set the `CHAP_API_TOKEN` environment variable on the Chap Core server. When it is unset,
authentication is disabled and the API behaves exactly as before.

Generate a token with any of these:

```console
openssl rand -hex 32
uuidgen
python -c "import secrets; print(secrets.token_hex(32))"
```

`uuidgen` is available on macOS and on Linux via `util-linux`, but is often missing from
slim container images -- `openssl` is the more portable option.

Use at least 32 characters. The API has no rate limiting, so a short token can be guessed by
anyone who can reach the port. Chap Core logs a warning at startup if the token is shorter
than that, but still starts.

With Docker Compose, add it to your `.env` file and it is passed through to the `chap`
service:

```console
CHAP_API_TOKEN=your-generated-token
```

Then restart the stack:

```console
docker compose up -d
```

## Making authenticated requests

Send the token as a bearer token in the `Authorization` header:

```console
curl -H "Authorization: Bearer your-generated-token" \
  http://localhost:8000/v1/crud/datasets
```

## Which endpoints require the token

When `CHAP_API_TOKEN` is set, **every** endpoint requires it except the three below. Note
that this includes the interactive documentation, so `/docs`, `/redoc` and `/openapi.json`
are not browsable on a token-protected instance.

| Path | Auth | Why |
|------|------|-----|
| `/health` | Public | Liveness probe. Called without headers by the container healthcheck and Kubernetes. |
| `/health/ready` | Public | Readiness probe, same reason. |
| `/system/info` | Public | Lets clients discover whether a token is required before authenticating. |
| Everything else | Required | Includes `/v1/**`, `/v2/**`, `/docs`, `/redoc` and `/openapi.json`. |

Service registration is covered too, with one accommodation for chapkit -- see
[Relation to the service registration key](#relation-to-the-service-registration-key).

## Error responses

When `CHAP_API_TOKEN` is configured:

| Scenario | Response |
|----------|----------|
| Missing `Authorization` header | 401 Unauthorized |
| Wrong token, or a scheme other than `Bearer` | 401 Unauthorized |

Both cases return the same body and a `WWW-Authenticate: Bearer` header, so a client can
tell "this server wants a token" apart from a network failure:

```json
{"detail": "Missing or invalid API token"}
```

## Detecting whether a server requires a token

`/system/info` stays public and reports `auth_required`, so a client can check before it has
a token:

```console
curl http://localhost:8000/system/info
```

```json
{
    "chap_core_version": "2.2.0",
    "python_version": "3.13.0",
    "server_date": "2026-01-01T12:00:00+00:00",
    "server_time_zone_id": "Etc/UTC",
    "revision": "",
    "auth_required": true
}
```

## Connecting from the Modeling App

The Modeling App reaches Chap Core through a DHIS2 Route. Configure the route to add the
`Authorization` header so the token is stored in DHIS2 rather than in the browser. See the
[DHIS2 Route API documentation](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-242/route.html)
for the `headers` and `auth` fields.

## Relation to the service registration key

`CHAP_API_TOKEN` and `SERVICEKIT_REGISTRATION_KEY` are separate secrets, configured and
checked independently, and either can be enabled without the other. See
[Service Registration](service-registration.md).

Self-registering chapkit services need special handling because servicekit can only send the
`X-Service-Key` header -- it has no way to send `Authorization`. The API token is therefore
accepted in `X-Service-Key` as well, on any path, and on the `/v2/services` paths a
configured registration key is accepted there too:

| Server configuration | What a chapkit service sends |
|----------------------|------------------------------|
| Only `CHAP_API_TOKEN` | `SERVICEKIT_REGISTRATION_KEY` set to the API token |
| Only `SERVICEKIT_REGISTRATION_KEY` | `SERVICEKIT_REGISTRATION_KEY` set to the registration key |
| Both | `SERVICEKIT_REGISTRATION_KEY` set to either one |

A registration key is only accepted on `/v2/services` paths. It cannot be used as a
general-purpose credential against `/v1/**`, so a model service cannot read or delete
datasets with it.

Other clients -- the Modeling App, `curl`, the Python SDK -- should use
`Authorization: Bearer`.
