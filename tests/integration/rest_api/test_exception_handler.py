import asyncio
import json
import logging

import pytest
from starlette.requests import Request

from chap_core.rest_api.app import global_exception_handler


@pytest.mark.parametrize("language", ["en", "fr"])
def test_unhandled_exception_returns_only_a_trace_id(caplog, french_catalog, language):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/boom",
            "headers": [(b"accept-language", language.encode())],
            "query_string": b"",
        }
    )

    with caplog.at_level(logging.ERROR, logger="chap_core.rest_api.app"):
        response = asyncio.run(global_exception_handler(request, RuntimeError("db password is hunter2")))

    body = json.loads(response.body)
    assert response.status_code == 500
    assert body.keys() == {"detail", "trace_id"}
    assert body["detail"] == (french_catalog["Internal server error"] if language == "fr" else "Internal server error")
    assert "hunter2" not in response.body.decode()
    assert body["trace_id"] in caplog.text
    assert "hunter2" in caplog.text
