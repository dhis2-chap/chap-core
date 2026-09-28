"""Thin client for the parts of the CHAP REST API that chap-admin drives.

Kept REST-only on purpose: it works against a remote instance and cannot drift from the
API the way direct database access would.
"""

import os
from typing import Any, cast

import httpx

DEFAULT_URL = "http://localhost:8000"


class ChapApiError(ValueError):
    """CHAP could not be reached or refused the request."""


class ChapApi:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 30):
        self.url = (url or os.getenv("CHAP_URL") or DEFAULT_URL).rstrip("/")
        token = token or os.getenv("CHAP_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ChapApi":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _call(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ChapApiError(f"Could not reach CHAP at {self.url}: {error}") from error
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ChapApiError(f"CHAP at {self.url} answered {response.status_code} for {method} {path}: {detail}")
        return response.json() if response.content else None

    def model_templates(self) -> list[dict[str, Any]]:
        return cast("list[dict[str, Any]]", self._call("GET", "/v1/crud/model-templates"))

    def create_model_template_from_service(self, service_id: str) -> dict[str, Any]:
        response = self._call("POST", "/v1/crud/model-templates/from-service", json={"service_id": service_id})
        return cast("dict[str, Any]", response)

    def archive_model_template(self, model_template_id: int, all_versions: bool = False) -> None:
        params = {"allVersions": "true"} if all_versions else None
        self._call("DELETE", f"/v1/crud/model-templates/{model_template_id}", params=params)

    def runnable_model_templates(self) -> list[dict[str, Any]]:
        """The live templates that are not retired. The listing also returns retired ones."""
        return [template for template in self.model_templates() if not template["archived"]]

    def create_configured_model(self, configured_model: dict[str, Any]) -> dict[str, Any]:
        return cast("dict[str, Any]", self._call("POST", "/v1/crud/configured-models", json=configured_model))

    def services(self) -> list[dict[str, Any]]:
        return cast("list[dict[str, Any]]", self._call("GET", "/v2/services")["services"])

    def deregister_service(self, service_id: str) -> None:
        """Drop a service's registration. Needs SERVICEKIT_REGISTRATION_KEY when the deployment sets one."""
        key = os.getenv("SERVICEKIT_REGISTRATION_KEY")
        headers = {"X-Service-Key": key} if key else {}
        try:
            self._call("DELETE", f"/v2/services/{service_id}", headers=headers)
        except ChapApiError as error:
            if "answered 404" not in str(error):
                raise
