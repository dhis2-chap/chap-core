# Servicekit Bugs

## SERVICEKIT_HOST env var not respected

**Severity**: Medium
**Found**: 2026-04-08

### Description
`SERVICEKIT_HOST` environment variable is ignored when set. The service always uses auto-detected hostname (e.g. `mlaptop.local`) regardless of the env var value.

Logs show `host_source=auto-detected` even when `SERVICEKIT_HOST=host.docker.internal` is exported before starting the service. The `SERVICEKIT_PORT` env var works correctly (`port_source=env:SERVICEKIT_PORT`).

### Impact
When chap-core runs in Docker and the chapkit service runs on the host, the auto-detected hostname is not resolvable from inside the Docker container. This causes:
- Config sync failure (`list_configs()` can't reach the service)
- No configured model created in chap-core
- Model doesn't appear in the modeling app

### Workaround
Pass `host` explicitly in `.with_registration()`:
```python
.with_registration(host="host.docker.internal")
```

### Expected behavior
Setting `SERVICEKIT_HOST=host.docker.internal` should override the auto-detected hostname.

### Location
`servicekit/src/servicekit/api/service_builder.py` - `with_registration()` method, host resolution logic around line 133.
