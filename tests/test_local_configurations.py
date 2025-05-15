

from chap_core.models.local_configuration import parse_local_model_config_file


def test_parse_local_model_config_file(data_path):
    # only tests that parsing of the default.yaml config file does not fail
    default_file = data_path.parent / 'config' / 'models' / 'default.yaml'
    assert default_file.exists()
    configurations = parse_local_model_config_file(default_file)


    
