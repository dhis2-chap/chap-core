import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints._common import load_data_configuration, load_dataset_from_csv
from chap_core.cli_endpoints.ensemble import evaluate_ensemble
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.cli_endpoints.validate import validate_cmd
from chap_core.database.dataset_tables import MaskedRange


@pytest.fixture
def laos_csv(data_path):
    """Monthly data for three provinces, 2010-01 to 2012-12."""
    return data_path / "laos_subset.csv"


@pytest.fixture
def mask_file(tmp_path):
    def _write(start: str, end: str, suffix: str = ".json"):
        path = tmp_path / f"data{suffix}"
        if suffix == ".json":
            path.write_text(json.dumps({"maskedRanges": [{"start": start, "end": end}]}))
        else:
            path.write_text(f'maskedRanges:\n  - start: "{start}"\n    end: "{end}"\n')
        return path

    return _write


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_data_configuration_file_is_read_as_json_or_yaml(mask_file, suffix):
    config = load_data_configuration(mask_file("2011-6", "2011-08", suffix))
    assert config is not None
    assert config.masked_ranges == [MaskedRange(start="201106", end="201108")]


def test_no_data_configuration_file_gives_none():
    assert load_data_configuration(None) is None


def test_data_configuration_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"maskedRanges": [], "columnMapping": {}}))
    with pytest.raises(ValidationError):
        load_data_configuration(path)


def test_csv_is_masked_with_the_data_configuration(laos_csv, mask_file):
    config = load_data_configuration(mask_file("2011-06", "2011-08"))
    dataset = load_dataset_from_csv(laos_csv, data_configuration=config)
    for data in dataset.values():
        assert np.all(np.isnan(data.disease_cases[17:20]))
        assert list(data.target_masked[16:21]) == [0, 1, 1, 1, 0]


def test_csv_with_a_mask_indicator_column_is_rejected(laos_csv, tmp_path):
    df = pd.read_csv(laos_csv)
    df["target_masked"] = 0
    csv = tmp_path / "with_indicator.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="reserved"):
        load_dataset_from_csv(csv)


def test_eval_rejects_a_mask_in_the_evaluation_window(laos_csv, mask_file, models_path, tmp_path):
    with pytest.raises(ValueError, match="Masked period 201211 falls in the evaluation window"):
        eval_cmd(
            model_name=str(models_path / "naive_python_model_with_mlproject_file"),
            dataset_csv=laos_csv,
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(ignore_environment=True),
            data_configuration=mask_file("2012-11", "2012-12"),
        )


def test_eval_runs_on_a_masked_csv(laos_csv, mask_file, models_path, tmp_path):
    output_file = tmp_path / "out.nc"
    eval_cmd(
        model_name=str(models_path / "naive_python_model_with_mlproject_file"),
        dataset_csv=laos_csv,
        output_file=output_file,
        backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
        run_config=RunConfig(ignore_environment=True),
        data_configuration=mask_file("2011-06", "2011-08"),
    )
    assert output_file.exists()


def test_evaluate_ensemble_rejects_a_data_configuration_with_a_named_dataset(mask_file, tmp_path):
    with pytest.raises(ValueError, match="--data-configuration needs --dataset-csv"):
        evaluate_ensemble(
            base_model_names="a,b",
            dataset_name="hydromet_5_filtered",
            report_filename=str(tmp_path / "report.pdf"),
            data_configuration=mask_file("2011-06", "2011-08"),
        )


def test_validate_reports_a_mask_of_the_wrong_period_type(laos_csv, mask_file, capsys):
    with pytest.raises(SystemExit):
        validate_cmd(dataset_csv=laos_csv, data_configuration=mask_file("2011W1", "2011W3"))
    assert "is a week range, but the dataset has month periods" in capsys.readouterr().out
