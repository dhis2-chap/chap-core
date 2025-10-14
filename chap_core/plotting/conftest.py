import pathlib

import pandas as pd
import pytest


@pytest.fixture()
def df() -> pd.DataFrame:
    path = pathlib.Path(__file__).parent.parent.parent / "example_data" / "monthly_data.csv"
    return pd.read_csv(path)
