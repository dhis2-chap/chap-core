import pandas as pd

from chap_core.file_io.external_file import get_file, fetch_and_clean
import pytest


@pytest.mark.skip(reason="Failing on CI")
def test_fetch_data():
    url = "https://github.com/drrachellowe/hydromet_dengue/raw/main/data/data_2000_2019.csv"
    file = get_file(url)


@pytest.mark.skip(reason="Failing on CI")
def test_fetch_and_clean(data_path):
    dataset = fetch_and_clean("hydromet")
    assert dataset is not None
    df: pd.DataFrame = dataset.to_pandas()
    df.to_csv(data_path / "hydromet.csv")
