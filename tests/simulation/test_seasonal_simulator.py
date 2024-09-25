from chap_core.simulation.seasonal_simulator import SeasonalSingleVariableSimulator


def test_seasonal_single_variable_simulator():
    """Test the SeasonalSingleVariableSimulator class."""
    simulator = SeasonalSingleVariableSimulator(
        n_seasons=5,
        n_data_points_per_season=10,
        mean_peak_height=100,
        peak_height_sd=10,
    )
    data = simulator.simulate()
    assert len(data) == 50
