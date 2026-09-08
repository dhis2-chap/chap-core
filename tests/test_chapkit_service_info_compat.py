"""The service info parser must accept fields added by newer chapkit services."""

from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config
from chap_core.rest_api.services.schemas import MLServiceInfo

CHAPKIT_2_INFO = {
    "id": "chapkit-ewars-model",
    "display_name": "CHAP-EWARS Model (chapkit)",
    "version": "1.0.0",
    "description": None,
    "git_revision": "fa880a1d8621c6c5bf60c472c299c40b5568ecd0",
    "chapkit_version": "2.0.0",
    "servicekit_version": "2.0.2",
    "model_metadata": {"author": "CHAP team", "author_assessed_status": "orange"},
    "period_type": "monthly",
    "min_prediction_periods": 0,
    "max_prediction_periods": 100,
    "allow_free_additional_continuous_covariates": True,
    "required_covariates": ["population"],
    "requires_geo": False,
    "some_future_field": {"nested": True},
}


def test_service_info_accepts_provenance_and_unknown_fields():
    info = MLServiceInfo.model_validate(CHAPKIT_2_INFO)

    assert info.git_revision == CHAPKIT_2_INFO["git_revision"]
    assert info.chapkit_version == "2.0.0"
    assert info.servicekit_version == "2.0.2"
    assert info.required_covariates == ["population"]


def test_service_info_from_chapkit_1_service_still_validates():
    payload = {
        k: v for k, v in CHAPKIT_2_INFO.items() if k not in ("git_revision", "chapkit_version", "servicekit_version")
    }
    info = MLServiceInfo.model_validate(payload)

    assert info.git_revision is None


def test_provenance_fields_do_not_break_template_conversion():
    info = MLServiceInfo.model_validate(CHAPKIT_2_INFO)
    config = ml_service_info_to_model_template_config(info, "http://ewars:8000")

    assert config.name == "chapkit-ewars-model"
    assert config.required_covariates == ["population"]
