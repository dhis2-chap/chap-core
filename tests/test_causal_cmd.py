import pandas as pd
import pytest

from chap_core.cli_endpoints.causal import _validate_datasets


def _make_df(locations, periods, extra_col_val=1.0):
    rows = [
        {"location": loc, "time_period": p, "rainfall": extra_col_val, "disease_cases": 0.0}
        for loc in locations
        for p in periods
    ]
    return pd.DataFrame(rows)


def test_validation_different_time_periods():
    original = _make_df(["A"], ["2022-01", "2022-02"])
    cf = _make_df(["A"], ["2022-01", "2022-03"])
    cf.loc[cf["time_period"] == "2022-03", "rainfall"] = 2.0
    with pytest.raises(ValueError, match="same time periods"):
        _validate_datasets(original, cf, ["rainfall"])


def test_validation_missing_column():
    original = _make_df(["A"], ["2022-01"])
    cf = _make_df(["A"], ["2022-01"])
    cf.loc[:, "rainfall"] = 2.0
    with pytest.raises(ValueError, match="not found"):
        _validate_datasets(original, cf, ["nonexistent_col"])


def test_validation_no_differences():
    original = _make_df(["A"], ["2022-01", "2022-02"])
    cf = original.copy()
    with pytest.raises(ValueError, match="No differences"):
        _validate_datasets(original, cf, ["rainfall"])


def test_validation_no_differences_row_order_independent():
    original = _make_df(["A"], ["2022-01", "2022-02"])
    cf = original.iloc[::-1].reset_index(drop=True)  # same data, reversed row order
    with pytest.raises(ValueError, match="No differences"):
        _validate_datasets(original, cf, ["rainfall"])


@pytest.mark.slow
def test_causal_cmd_integration(tmp_path):
    import pandas as pd

    from chap_core.api_types import RunConfig
    from chap_core.cli_endpoints.causal import causal_cmd
    from chap_core.file_io.example_data_set import datasets

    dataset = datasets["hydromet_5_filtered"].load()
    original_csv = tmp_path / "original.csv"
    dataset.to_csv(original_csv)

    # Build counterfactual by modifying the rainfall column
    df = pd.read_csv(original_csv)
    df["rainfall"] = df["rainfall"] + 10.0
    cf_csv = tmp_path / "counterfactual.csv"
    df.to_csv(cf_csv, index=False)

    # Leave a small prediction window (last 3 periods); read strings from CSV
    # to avoid dealing with TimePeriod.__str__ giving 'Month(2019-10)' instead of '2019-10'
    periods = sorted(df["time_period"].unique())
    split_period = periods[-3]

    output_file = tmp_path / "causal_out.nc"
    causal_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=str(original_csv),
        counterfactual_csv=str(cf_csv),
        counterfactual_columns=["rainfall"],
        split_period=split_period,
        output_file=output_file,
        run_config=RunConfig(),
    )

    assert output_file.exists(), "Original predictions NetCDF not created"
    cf_output = tmp_path / "causal_out_cf.nc"
    assert cf_output.exists(), "Counterfactual predictions NetCDF not created"
