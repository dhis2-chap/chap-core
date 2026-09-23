import pytest

from chap_core.models.local_configuration import (
    parse_local_model_config_file,
    parse_local_model_config_from_directory,
    LocalModelTemplateWithConfigurations,
    MarketplaceModelSeed,
)


def test_parse_local_model_config_file(data_path):
    # only tests that parsing of the default.yaml config file does not fail
    default_file = data_path.parent / "config" / "configured_models" / "default.yaml"
    assert default_file.exists()
    configurations = parse_local_model_config_file(default_file)
    print(configurations)


def test_parse_local_model_config_files_from_directory(data_path):
    default_directory = data_path.parent / "config" / "configured_models"
    configurations = parse_local_model_config_from_directory(
        default_directory, search_pattern="*.yaml"
    )  # TODO: should we also test the .disabled yamls?
    assert isinstance(configurations, list)
    assert len(configurations) > 1
    assert isinstance(configurations[0], LocalModelTemplateWithConfigurations)


def _git_model_config(tmp_path, middle_ref):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        "- url: https://github.com/example/model\n"
        "  versions:\n"
        f'    v1: "@{"a" * 40}"\n'
        f'    v2: "{middle_ref}"\n'
        f'    v3: "@{"b" * 40}"\n'
    )
    return config_file


@pytest.mark.parametrize("ref", ["@main", "main", "@v1.2.0", "@abc1234", "abc1234def"])
def test_git_version_must_be_a_full_commit_sha_under_every_label(tmp_path, ref):
    """A moving ref is rejected when the file is parsed, even under a label that is not seeded."""
    config_file = _git_model_config(tmp_path, ref)
    with pytest.raises(ValueError, match=f"version 'v2' of https://github.com/example/model is '{ref}'") as e:
        parse_local_model_config_file(config_file)
    assert str(config_file) in str(e.value)


def test_git_version_accepts_full_commit_sha_with_or_without_at(tmp_path):
    configuration = parse_local_model_config_file(_git_model_config(tmp_path, "c" * 40))[0]
    assert isinstance(configuration, LocalModelTemplateWithConfigurations)
    assert list(configuration.versions.values()) == ["@" + "a" * 40, "c" * 40, "@" + "b" * 40]


def test_marketplace_entries_are_parsed_next_to_git_entries(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        "- marketplace: chapkit_ewars_model\n"
        "- url: https://github.com/example/model\n"
        "  versions:\n"
        f'    v1: "@{"a" * 40}"\n'
    )
    marketplace_seed, repository = parse_local_model_config_file(config_file)
    assert isinstance(marketplace_seed, MarketplaceModelSeed)
    assert marketplace_seed.marketplace == "chapkit_ewars_model"
    assert isinstance(repository, LocalModelTemplateWithConfigurations)


def test_chapkit_versions_are_not_required_to_be_commit_shas(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text('- url: http://model:8000\n  uses_chapkit: true\n  versions:\n    v1: "/v1"\n')
    configuration = parse_local_model_config_file(config_file)[0]
    assert isinstance(configuration, LocalModelTemplateWithConfigurations)
    assert configuration.uses_chapkit is True
