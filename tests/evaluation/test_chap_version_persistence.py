"""A backtest imported from a .nc file records the version of the writer, not of the importer."""

import xarray as xr

import chap_core
from chap_core.assessment.evaluation import Evaluation


def _write_with_version(backtest, tmp_path, chap_version):
    """Write the backtest to a file whose chap_version attr is rewritten; None removes it."""
    filepath = tmp_path / "eval.nc"
    Evaluation.from_backtest(backtest).to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")
    ds = xr.open_dataset(filepath).load()
    if chap_version is None:
        del ds.attrs["chap_version"]
    else:
        ds.attrs["chap_version"] = chap_version
    rewritten = tmp_path / "rewritten.nc"
    ds.to_netcdf(rewritten)
    return rewritten


def test_from_file_records_the_version_of_the_writer(backtest, tmp_path):
    filepath = _write_with_version(backtest, tmp_path, "9.9.9")

    assert Evaluation.from_file(filepath).to_backtest().chap_version == "9.9.9"


def test_from_file_round_trips_this_checkouts_version(backtest, tmp_path):
    filepath = tmp_path / "eval.nc"
    Evaluation.from_backtest(backtest).to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")

    expected = None if chap_core.__version__ == "unknown" else chap_core.__version__
    assert Evaluation.from_file(filepath).to_backtest().chap_version == expected


def test_from_file_gives_none_when_the_file_has_no_version(backtest, tmp_path):
    filepath = _write_with_version(backtest, tmp_path, None)

    assert Evaluation.from_file(filepath).to_backtest().chap_version is None
