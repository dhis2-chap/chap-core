import pytest
import httpx
from chap_core.models.external_chapkit_model import ExternalChapkitModel, ExternalChapkitModelTemplate
from chap_core.file_io.example_data_set import datasets

model_url = "http://localhost:8001"

@pytest.fixture
def dataset():
    dataset_name = "ISIMIP_dengue_harmonized"
    dataset = datasets[dataset_name]
    dataset = dataset.load()
    dataset = dataset["brazil"]
    return dataset



@pytest.fixture
def service_available():
    try:
        response = httpx.get(model_url + "/api/v1/health", timeout=2)
        if response.status_code != 200:
            pytest.skip("Service not available at localhost:8001")
    except:
        pytest.skip("Service not available at localhost:8001")

    return model_url


def test_external_chapkit_model_basic(service_available, dataset):
    template = ExternalChapkitModelTemplate("example_model", service_available)
    model = template.get_model({})
    id = model.train(dataset)
    print(id)

