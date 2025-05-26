

from chap_core.models.local_configuration import parse_local_model_config_file, parse_local_model_config_from_directory, LocalModelTemplateWithConfigurations


def test_parse_local_model_config_file(data_path):
    # only tests that parsing of the default.yaml config file does not fail
    default_file = data_path.parent / 'config' / 'configured_models' / 'default.yaml'
    assert default_file.exists()
    configurations = parse_local_model_config_file(default_file)
    print(configurations)


def test_parse_local_model_config_files_from_directory(data_path):
    default_directory = data_path.parent / 'config' / 'configured_models'
    configurations = parse_local_model_config_from_directory(default_directory, search_pattern="*.yaml") # TODO: should we also test the .disabled yamls? 
    assert isinstance(configurations, list)
    assert len(configurations) > 1
    assert isinstance(configurations[0], LocalModelTemplateWithConfigurations)
    

    
