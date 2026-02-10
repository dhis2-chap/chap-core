from chap_core.api_types import FeatureModel


def test_feature_model():
    txt = """{
    "id": "l3Lcy6a8hNq",
    "type": "Feature",
    "properties": {
        "id": "l3Lcy6a8hNq",
        "parent": "hdeC7uX9Cko",
        "parentGraph": "hdeC7uX9Cko",
        "level": 3
    }
}"""
    FeatureModel.model_validate_json(txt)
