from chap_core.assessment.flat_representations import FlatForecasts
import pandas as pd


def test_forecast_schema_acceptance():
    data = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc2", "loc2"],
            "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
            "horizon_distance": [1, 2, 1, 2],
            "sample": [1, 1, 1, 1],
            "forecast": [10.0, 12.0, 20.0, 22.0],
        }
    )

    data = FlatForecasts(data)
    print(data)
