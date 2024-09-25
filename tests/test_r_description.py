from chap_core.external.r_description import get_imports


def test_parse(data_path):
    # Example usage
    file_path = data_path / "DESCRIPTION"
    imports = get_imports(file_path)
    assert imports == ["dplyr", "tidyr"]
