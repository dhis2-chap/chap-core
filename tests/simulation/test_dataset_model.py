import pytest

from chap_core.simulation.naive_simulator import DatasetDimensions, AdditiveSimulator, BacktestSimulator


@pytest.fixture
def dims():
    return DatasetDimensions(
        locations=["loc1", "loc2", "loc3"],
        time_periods=[f"{year}-{month:02d}" for year in ("2020", "2021", "2022") for month in range(1, 13)],
        target="disease_cases",
        features=[],
    )


def test_additive_simulator(dims):
    simulator = AdditiveSimulator()
    dataset = simulator.simulate(dims)
    print(dataset.observations)


def test_forecast_simulator(dims):
    dataset = AdditiveSimulator().simulate(dims)
    backtest = BacktestSimulator().simulate(dataset, dims)
    assert len(backtest.forecasts) == len(dims.locations) * 3 * 2
