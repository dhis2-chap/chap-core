from climate_health.main import assess_model_on_csv_data
from climate_health.predictor.poisson import Poisson
from climate_health.simulation.random_noise_simulator import RandomNoiseSimulator


def test_simulation_integration(tmp_path):
    data = RandomNoiseSimulator(100).simulate()
    file_name = tmp_path / "data.csv"
    data.to_csv(file_name)
    report = assess_model_on_csv_data(
        data_file_name=file_name, split_fraction=0.7, model=Poisson()
    )
    print(report.rmse_dict)
