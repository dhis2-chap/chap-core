import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from chap_core.hpo.search_space import (
    Float,
    Int,
    search_space_from_config,
    serialize_search_space,
    validate_search_space,
)


def test_search_space_from_yaml_parses_all_supported_types(tmp_path: Path) -> None:
    """A YAML HPO file is parsed into categorical, integer, continuous and stepped runtime domains."""
    yml = tmp_path / "space.yaml"
    yml.write_text(
        """
weight_decay:
  type: float
  low: 1.0e-6
  high: 1.0e-5
  log: true
max_epochs:
  type: int
  low: 1
  high: 3
  log: false
  step: 1
learning_rate:
  values: [1.0e-3, 1.0e-2]
batch_size:
  values: [64, 32]
augmentations:
  values: [[]]
context_length:
  values: [12]
past_ratio:
  type: float
  low: 0.2
  high: 0.4
  step: 0.1
        """.strip(),
        encoding="utf-8",
    )

    with yml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    space = search_space_from_config(config)

    assert space["learning_rate"] == [1e-3, 1e-2]
    assert space["batch_size"] == [64, 32]
    assert space["augmentations"] == [[]]
    assert space["context_length"] == [12]
    assert space["weight_decay"] == Float(low=1e-6, high=1e-5, step=None, log=True)
    assert space["past_ratio"] == Float(low=0.2, high=0.4, step=0.1, log=False)
    assert space["max_epochs"] == Int(low=1, high=3, step=1, log=False)


def test_search_space_from_config_parses_categorical_int_and_float_specs() -> None:
    """YAML-shaped HPO configuration is converted into the runtime domain objects searchers consume."""
    config = {
        "family": {"values": ["linear", "tree"]},
        "depth": {"low": 1, "high": 7, "step": 2},
        "workers": {"type": "int", "low": 1, "high": 16, "log": True},
        "learning_rate": {"low": 0.01, "high": 0.2},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    }

    space = search_space_from_config(config)

    assert space == {
        "family": ["linear", "tree"],
        "depth": Int(low=1, high=7, step=2, log=False),
        "workers": Int(low=1, high=16, step=1, log=True),
        "learning_rate": Float(low=0.01, high=0.2, step=None, log=False),
        "dropout": Float(low=0.0, high=0.5, step=0.1, log=False),
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param({"x": 1}, "each spec must be a mapping", id="non-mapping"),
        pytest.param({"x": {"values": []}}, "'values' must be a non-empty list", id="empty-values"),
        pytest.param({"x": {"values": "a"}}, "'values' must be a non-empty list", id="values-not-list"),
        pytest.param({"x": {"low": 1}}, "expected 'low' and 'high'", id="missing-high"),
        pytest.param(
            {"x": {"type": "int", "low": 1, "high": 10, "step": 2, "log": True}},
            "log-int requires step==1",
            id="log-int-step",
        ),
        pytest.param(
            {"x": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1, "log": True}},
            "log-float requires step==None",
            id="log-float-step",
        ),
        pytest.param(
            {"x": {"type": "decimal", "low": 0, "high": 1}},
            "unknown spec type",
            id="unknown-type",
        ),
    ],
)
def test_search_space_from_config_rejects_malformed_specs(
    config: dict[str, Any],
    message: str,
) -> None:
    """Configuration parsing reports structural YAML errors before runtime-domain validation begins."""
    with pytest.raises(ValueError, match=message):
        search_space_from_config(config)


def test_validate_search_space_returns_independent_categorical_lists() -> None:
    """Validation normalizes domains without retaining ownership of caller-owned categorical lists."""
    values = ["a", "b"]
    validated = validate_search_space({"choice": values, "depth": Int(1, 3)})

    values.append("c")

    assert validated["choice"] == ["a", "b"]
    assert validated["choice"] is not values
    assert validated["depth"] == Int(1, 3)


def test_serialize_search_space_produces_json_safe_explicit_specs() -> None:
    """Runtime Int/Float domains become explicit JSON-safe metadata while categorical values are preserved."""
    serialized = serialize_search_space(
        {
            "category": ["a", "b"],
            "depth": Int(1, 7, step=2),
            "rate": Float(1e-3, 1.0, log=True),
        }
    )

    assert serialized == {
        "category": ["a", "b"],
        "depth": {"type": "int", "low": 1, "high": 7, "step": 2, "log": False},
        "rate": {"type": "float", "low": 1e-3, "high": 1.0, "step": None, "log": True},
    }
    json.dumps(serialized)
