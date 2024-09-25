import os
import tempfile

import pytest

from typing import Annotated, Generator
from chap_core.omnipy_lib import StrDataset, IsRuntime, Runtime
from pytest import fixture

from chap_core.data_wrangling.tasks import load_data_as_clean_strings
from chap_core.data_wrangling.flows import (
    standardize_separated_data_func_flow,
    standardize_separated_data_linear_flow,
    standardize_separated_data_dag_flow,
    standardize_separated_data_wrapper,
)

from .. import EXAMPLE_DATA_PATH


@pytest.fixture(scope="module")
def tmp_dir_path() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as _tmp_dir_path:
        yield _tmp_dir_path


@pytest.fixture(scope="module")
def runtime(
    tmp_dir_path: Annotated[str, pytest.fixture],
) -> Generator[IsRuntime, None, None]:
    runtime = Runtime()

    runtime.config.job.output_storage.local.persist_data_dir_path = os.path.join(
        tmp_dir_path, "outputs"
    )
    runtime.config.root_log.file_log_dir_path = os.path.join(tmp_dir_path, "logs")

    yield runtime


@fixture(scope="module")
def separated_data(
    runtime: Annotated[Generator[IsRuntime, None, None], pytest.fixture],
) -> Annotated[Generator[StrDataset, None, None], pytest.fixture]:
    separate_data_path = EXAMPLE_DATA_PATH / "nonstandard_separate"

    ds = load_data_as_clean_strings.run(str(separate_data_path))
    yield ds
    os.unlink(f"{separate_data_path}.tar.gz")


@fixture(scope="module")
def separate_standardized_path() -> (
    Annotated[Generator[str, None, None], pytest.fixture]
):
    separate_standardized_path = str(
        EXAMPLE_DATA_PATH / "nonstandard_separate_standardized"
    )
    yield separate_standardized_path
    os.unlink(f"{separate_standardized_path}.tar.gz")


def test_load_separated_data(
    separated_data: Annotated[Generator[StrDataset, None, None], pytest.fixture],
):
    assert isinstance(separated_data, StrDataset)
    assert len(separated_data) == 3
    assert tuple(separated_data.keys()) == (
        "separated_disease_data",
        "separated_rain_data",
        "separated_temp_data",
    )
    assert separated_data["separated_disease_data"].startswith("periodname")


@pytest.mark.skip(
    reason="Management of keyword arguments for linear flows in Omnipy are suboptimal, needs to be "
    "improved. In any case redundant with FuncFlow"
)
def test_standardize_separated_data_linear_flow(
    separated_data: Annotated[StrDataset, pytest.fixture],
    separate_standardized_path: Annotated[str, pytest.fixture],
):
    standardized_table_ds = standardize_separated_data_wrapper.run(
        standardize_separated_data_linear_flow,
        separated_data,
        time_period_col_name="periodname",
        rain_data_file_name="separated_rain_data",
        temp_data_file_name="separated_temp_data",
        disease_data_file_name="separated_disease_data",
    )
    standardized_table_ds.save(separate_standardized_path)


@pytest.mark.skip(
    reason="Depends on planned modifiers that are not yet implemented in Omnipy. "
    "In any case redundant with FuncFlow"
)
def test_standardize_separated_data_dag_flow(
    separated_data: Annotated[StrDataset, pytest.fixture],
    separate_standardized_path: Annotated[str, pytest.fixture],
):
    standardized_table_ds = standardize_separated_data_wrapper.run(
        standardize_separated_data_dag_flow,
        separated_data,
        time_period_col_name="periodname",
        rain_data_file_name="separated_rain_data",
        temp_data_file_name="separated_temp_data",
        disease_data_file_name="separated_disease_data",
    )
    standardized_table_ds.save(separate_standardized_path)


def test_standardize_separated_data_func_flow(
    separated_data: Annotated[StrDataset, pytest.fixture],
    separate_standardized_path: Annotated[str, pytest.fixture],
):
    standardized_table_ds = standardize_separated_data_func_flow.run(
        separated_data,
        time_period_col_name="periodname",
        rain_data_file_name="separated_rain_data",
        temp_data_file_name="separated_temp_data",
        disease_data_file_name="separated_disease_data",
    )
    standardized_table_ds.save(separate_standardized_path)


def test_standardize_separated_data_func_flow_refined_variant(
    separated_data: Annotated[StrDataset, pytest.fixture],
    separate_standardized_path: Annotated[str, pytest.fixture],
):
    refined_standardize_separated_data = standardize_separated_data_func_flow.refine(
        name="refined_standardize_separated_data",
        fixed_params=dict(
            time_period_col_name="periodname",
            rain_data_file_name="separated_rain_data",
            temp_data_file_name="separated_temp_data",
            disease_data_file_name="separated_disease_data",
        ),
    )

    standardized_table_ds = refined_standardize_separated_data.run(separated_data)
    standardized_table_ds.save(separate_standardized_path)
