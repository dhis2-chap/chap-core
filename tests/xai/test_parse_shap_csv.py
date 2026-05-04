"""Tests for native SHAP CSV parsing and metadata building."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chap_core.models.external_model import _parse_shap_csv
from chap_core.rest_api.db_worker_functions import _build_native_shap_metadata
from chap_core.xai.method_registry import NATIVE_SHAP


def _write_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "shap_values.csv"
    df.to_csv(p, index=False)
    return p


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": ["A", "A", "B"],
            "time_period": ["2024-01", "2024-02", "2024-01"],
            "expected_value": [10.0, 10.0, 10.0],
            "shap__rainfall": [0.5, 0.6, -0.2],
            "shap__temp": [-0.3, -0.2, 0.4],
            "value__rainfall": [80.0, 91.0, 54.0],
            "value__temp": [28.0, 29.0, 31.0],
        }
    )


def test_parses_valid_csv(tmp_path: Path):
    p = _write_csv(tmp_path, _valid_df())
    out = _parse_shap_csv(p)
    assert out["feature_names"] == ["rainfall", "temp"]
    assert len(out["values"]) == 3
    assert out["values"][0]["feature_values"] == {"rainfall": 80.0, "temp": 28.0}
    assert out["values"][0]["shap_values"] == [0.5, -0.3]


def test_rejects_nan_in_shap_columns(tmp_path: Path):
    df = _valid_df()
    df.loc[0, "shap__rainfall"] = float("nan")
    p = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="NaN .* shap"):
        _parse_shap_csv(p)


def test_rejects_oversized_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHAP_NATIVE_SHAP_MAX_BYTES", "100")  # 100 bytes
    p = _write_csv(tmp_path, _valid_df())
    with pytest.raises(ValueError, match="exceeds"):
        _parse_shap_csv(p)


def test_rejects_too_many_features(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHAP_NATIVE_SHAP_MAX_FEATURES", "1")
    p = _write_csv(tmp_path, _valid_df())
    with pytest.raises(ValueError, match="too many"):
        _parse_shap_csv(p)


def test_rejects_malformed_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHAP_NATIVE_SHAP_MAX_BYTES", "abc")
    p = _write_csv(tmp_path, _valid_df())
    with pytest.raises(ValueError, match="CHAP_NATIVE_SHAP_MAX_BYTES"):
        _parse_shap_csv(p)


def test_rejects_non_numeric_shap_column(tmp_path: Path):
    df = _valid_df()
    df["shap__rainfall"] = df["shap__rainfall"].astype(object)
    df.loc[0, "shap__rainfall"] = "not-a-number"
    p = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="shap__rainfall"):
        _parse_shap_csv(p)


def _sample_native_shap() -> dict:
    return {
        "feature_names": ["rainfall", "temp"],
        "expected_value": 10.0,
        "values": [
            {"location": "A", "time_period": "2024-01", "shap_values": [0.5, -0.3], "expected_value": 10.0},
            {"location": "B", "time_period": "2024-01", "shap_values": [-0.2, 0.4], "expected_value": 10.0},
        ],
    }


def test_build_native_shap_metadata_excludes_raw_values():
    native_shap = _sample_native_shap()
    meta = _build_native_shap_metadata(native_shap)
    assert NATIVE_SHAP not in meta, "raw SHAP rows must not be stored in meta_data"


def test_build_native_shap_metadata_contains_global_summary():
    native_shap = _sample_native_shap()
    meta = _build_native_shap_metadata(native_shap)
    global_entry = meta["xai"]["global_by_method"][NATIVE_SHAP]
    assert global_entry["nSamples"] == 2
    assert len(global_entry["topFeatures"]) == 2


def test_build_native_shap_metadata_direction():
    native_shap = {
        "feature_names": ["rain", "temp"],
        "expected_value": 0.0,
        "values": [
            {"location": "A", "time_period": "2024-01", "shap_values": [1.0, -2.0]},
            {"location": "B", "time_period": "2024-01", "shap_values": [0.5, -1.0]},
        ],
    }
    meta = _build_native_shap_metadata(native_shap)
    features = {f["feature_name"]: f for f in meta["xai"]["global_by_method"][NATIVE_SHAP]["topFeatures"]}
    assert features["rain"]["direction"] == "positive"
    assert features["temp"]["direction"] == "negative"
