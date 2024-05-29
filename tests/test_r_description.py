from climate_health.external.r_description import parse_description_file, get_imports


def test_parse(data_path):
    # Example usage
    file_path = data_path / 'DESCRIPTION'
    imports = get_imports(file_path)
    assert imports == ['dplyr', 'tidyr']


