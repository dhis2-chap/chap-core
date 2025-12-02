import pytest
from chap_core.models.chap_user_options import ChapUserOptions


def test_chap_user_options_defaults():
    """Test that ChapUserOptions has correct default values."""
    options = ChapUserOptions()
    assert options.chap__covid_mask is False


def test_extract_from_config_empty():
    """Test extraction from empty config."""
    options = ChapUserOptions.extract_from_config({})
    assert options.chap__covid_mask is False


def test_extract_from_config_with_chap_options():
    """Test extraction with chap__ prefixed options."""
    config = {"chap__covid_mask": True, "other_option": "value"}
    options = ChapUserOptions.extract_from_config(config)
    assert options.chap__covid_mask is True


def test_extract_from_config_filters_non_chap():
    """Test that only chap__ prefixed options are extracted."""
    config = {"chap__covid_mask": True, "model_specific_param": 10, "another_param": "test"}
    options = ChapUserOptions.extract_from_config(config)
    assert options.chap__covid_mask is True


def test_to_json_schema_properties():
    """Test JSON schema generation."""
    options = ChapUserOptions()
    schema = options.to_json_schema_properties()

    assert "chap__covid_mask" in schema
    assert schema["chap__covid_mask"]["type"] == "boolean"
    assert schema["chap__covid_mask"]["default"] is False
    assert "description" in schema["chap__covid_mask"]


def test_extract_from_none():
    """Test extraction when user_option_values is None."""
    options = ChapUserOptions.extract_from_config(None)
    assert options.chap__covid_mask is False
