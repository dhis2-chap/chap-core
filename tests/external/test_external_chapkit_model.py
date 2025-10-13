from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.models.external_chapkit_model import ExternalChapkitModelTemplate
import pytest
import httpx

# from chap_core.models.external_chapkit_model import ExternalChapkitModel, ExternalChapkitModelTemplate
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
        response = httpx.get(model_url + "/health", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Service not available at {model_url}")
    except:
        pytest.skip(f"Service not available at {model_url}")

    return model_url


#@pytest.mark.skip(reason="Needs a running chapkit model service")
def test_external_chapkit_model_basic(service_available, dataset):
    train, test = train_test_generator(dataset, 3, 2)
    historic, future, truth = next(test)

    template = ExternalChapkitModelTemplate("example_model", service_available)
    model = template.get_model({"max_epochs": 2})
    # print(dataset)
    id = model.train(historic)
    prediction = model.predict(historic, future)
    print("PREDICTION")
    print(prediction)
    prediction = prediction.to_pandas()

    print(prediction)
    assert len(prediction) == len(future.to_pandas())
