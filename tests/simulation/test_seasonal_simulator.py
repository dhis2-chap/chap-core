from climate_health.simulation.seasonal_simulator import SeasonalSingleVariableSimulator


def test_seasonal_single_variable_simulator():
    """Test the SeasonalSingleVariableSimulator class."""
    simulator = SeasonalSingleVariableSimulator(
        n_seasons=5, n_data_points_per_season=10, max_peak_height=100
    )
    data = simulator.simulate()
    assert len(data) == 50
    assert data.max() <= 100
