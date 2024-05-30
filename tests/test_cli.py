from climate_health.api import forecast
import pytest
from climate_health.util import docker_available


@pytest.mark.skipif(not docker_available())
def test_forecast_github_model():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    results = forecast("external", "hydromet_5_filtered", 12, repo_url)
