from pathlib import Path, PurePath

from chap_core.external.external_model import get_model_from_yaml_file, DryModeExternalCommandLineModel, \
    VerboseRDryModeExternalCommandLineModel
from tests.external.test_external_models import get_dataset_from_yaml


def train_model_and_predict(model_directory='ewars_Plus', models_path = Path(__file__).parent.parent / 'external_models'):
    yaml_path = models_path / model_directory / 'config.yml'
    train_data = get_dataset_from_yaml(yaml_path)
    #model = VerboseRDryModeExternalCommandLineModel.from_yaml_file(yaml_path)
    model = DryModeExternalCommandLineModel.from_yaml_file(yaml_path)
    model.set_file_creation_folder(Path(__file__).parent / PurePath('drydata'))
    model.setup()
    model.train(train_data)
    try:
        results = model.predict(train_data)
    except ValueError:
        pass
    print(model.get_execution_code())

train_model_and_predict()
