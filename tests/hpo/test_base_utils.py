import io
import os
import yaml
import json
from pathlib import Path

from chap_core.hpo.base import Int, Float, dedup, write_yaml, load_search_space_from_config


def test_dedup_handles_scalars_lists_and_dicts():
    assert dedup(3) == [3]
    assert dedup([1, 1, 2, 2, 3]) == [1, 2, 3]
    # dicts and nested structures should dedupe by value
    vals = [{"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}, {"a": 2}]
    assert dedup(vals) == [{"a": 1, "b": [1, 2]}, {"a": 2}]
    # None and mixed
    assert dedup([None, None, 0, 0]) == [None, 0]


def test_write_yaml_roundtrips(tmp_path: Path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
    out = tmp_path / "out.yaml"
    write_yaml(str(out), data)
    assert out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert loaded == data


def test_load_search_space_from_yaml_parses_all_types(tmp_path: Path):
    yml = tmp_path / "space.yaml"
    yml.write_text(
        """
weight_decay: 
  type: float
  low: 1e-6
  high: 1e-5
  log: True
max_epochs:
  type: int
  low: 1
  high: 3
  log: False
  step: 1
learning_rate: 
  values: [1e-3, 1e-2]
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
        """.strip()
    )

    with open(yml, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    space = load_search_space_from_config(configs)
    # categorical
    assert isinstance(space["batch_size"], list) and space["batch_size"] == [64, 32]
    # Floats
    assert isinstance(space["weight_decay"], Float)
    assert isinstance(space["past_ratio"], Float) and space["past_ratio"].log is False
    # Ints
    assert isinstance(space["max_epochs"], Int)
    assert isinstance(space["max_epochs"], Int) and space["max_epochs"].step == 1


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__]))
