from typing import Any

from chap_core.rest_api.v1.xai_schemas import SurrogateQualityRead


def quality_response_dict(quality: dict[str, Any] | None) -> dict[str, Any] | None:
    if not quality:
        return None
    return SurrogateQualityRead.model_validate(quality).model_dump(by_alias=True)
