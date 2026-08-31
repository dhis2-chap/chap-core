import logging

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSet, DataSetCreateInfo
from chap_core.database.datasets_seed import seed_example_datasets
from chap_core.database.model_template_seed import (
    add_configured_model,
    add_model_template_from_url,
    seed_configured_models_from_config_dir,
)
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    drifted_template_content_fields,
    ModelConfiguration,
    ModelTemplateDB,
    ModelTemplateMetaData,
)
from chap_core.database.tables import Backtest
from chap_core.datatypes import HealthPopulationData
from chap_core.external.model_configuration import (
    CommandConfig,
    DockerEnvConfig,
    EntryPointConfig,
    ModelTemplateConfigV2,
)
from chap_core.models.external_model import ExternalModel
from chap_core.rest_api.data_models import BacktestCreate, ModelTemplateRead
from chap_core.rest_api.db_worker_functions import run_backtest, run_prediction
from chap_core.testing.testing import assert_dataset_equal

logger = logging.getLogger(__name__)


template_urls = [
    "https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a",
    "https://github.com/knutdrand/weekly_ar_model@15cc39068498a852771c314e8ea989e6b555b8a5",
    "https://github.com/dhis2-chap/chap_auto_ewars@0c41b1d9bd187521e62c58d581e6f5bd5127f7b5",
    "https://github.com/dhis2-chap/chap_auto_ewars_weekly@51c63a8581bc29bdb40e788a83f701ed30cca83f",
]


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def engine_with_dataset(engine, weekly_full_data):
    with SessionWrapper(engine) as session:
        DataSetManager(session.session).save_dataset(DataSetCreateInfo(name="full_data"), weekly_full_data, None)
    return engine


def test_dataset_roundrip(health_population_data, engine):
    info = DataSetCreateInfo(name="health_population")
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(info, health_population_data, None)
        dataset = DataSetManager(session.session).to_dataset(dataset_id, HealthPopulationData)
        assert_dataset_equal(dataset, health_population_data)


@pytest.mark.skip("Needs to seed models for this test to work")
def test_backtest(engine_with_dataset):
    with Session(engine_with_dataset) as session:
        dataset_id = session.exec(select(DataSet.id)).first()
    with SessionWrapper(engine_with_dataset) as session:
        res = run_backtest(BacktestCreate(model_id="naive_model", dataset_id=dataset_id), 12, 2, 1, session=session)
    with Session(engine_with_dataset) as session:
        backtests = session.exec(select(Backtest)).all()
        assert len(backtests) == 1
        backtest = backtests[0]
        assert backtest.dataset_id == dataset_id
        assert len(backtest.forecasts) == 12 * 2 * 10


@pytest.mark.skip("Needs to seed models for this test to work")
def test_add_predictions(engine_with_dataset):
    with SessionWrapper(engine_with_dataset) as session:
        run_prediction("naive_model", 1, 3, name="testing", session=session)


@pytest.fixture
def model_template_yaml_config():
    return ModelTemplateConfigV2(
        name="test_model",
        version="test-version",
        required_covariates=["rainfall", "mean_temperature"],
        allow_free_additional_continuous_covariates=False,
        user_options={},
        meta_data=ModelTemplateMetaData(
            author="chap_temp",
            author_note="Testing author note",
            author_assessed_status="green",
            description="my model",
            display_name="My Model",
        ),
        entry_points=EntryPointConfig(
            train=CommandConfig(command="train", parameters={"param1": "value1"}),
            predict=CommandConfig(command="predict", parameters={"param2": "value2"}),
        ),
        docker_env=DockerEnvConfig(image="my_docker_image"),
    )


def test_add_model_template_from_yaml_config(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        model_template = session.get_model_template(id)
        assert model_template.name == model_template_yaml_config.name
        assert model_template.required_covariates == model_template_yaml_config.required_covariates
        assert (
            model_template.allow_free_additional_continuous_covariates
            == model_template_yaml_config.allow_free_additional_continuous_covariates
        )
        assert model_template.user_options == model_template_yaml_config.user_options
        assert model_template.author_assessed_status == model_template_yaml_config.meta_data.author_assessed_status


def test_add_model_template_from_yaml_config_requires_version(model_template_yaml_config, engine):
    model_template_yaml_config.version = None

    with SessionWrapper(engine) as session, pytest.raises(ValueError, match="must declare a version"):
        session.add_model_template_from_yaml_config(model_template_yaml_config)


def test_model_template_read_accepts_null_version():
    read = ModelTemplateRead.model_validate({"name": "legacy", "id": 1, "version": None})
    assert read.version is None


def test_model_spec_read_accepts_null_version():
    read = ModelSpecRead.model_validate(
        {
            "name": "legacy",
            "id": 1,
            "version": None,
            "covariates": [],
            "target": {"name": "disease_cases", "displayName": "Disease cases", "description": "Disease cases"},
        }
    )
    assert read.version is None


def test_add_model_template_unarchives_existing(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        template = session.session.get(ModelTemplateDB, template_id)
        template.archived = True
        session.session.commit()

    with SessionWrapper(engine) as session:
        template = session.session.get(ModelTemplateDB, template_id)
        assert template.archived is True

    with SessionWrapper(engine) as session:
        returned_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        assert returned_id == template_id
        template = session.session.get(ModelTemplateDB, template_id)
        assert template.archived is False


def test_add_configured_model_chapkit_skips_required_validation(engine):
    # Mimics what chapkit's /api/v1/configs/$schema returns for a field declared
    # with Field(default_factory=lambda: [3]): the property has no literal
    # "default" key. chap-core's heuristic-based validator would mark it
    # required, but with uses_chapkit=True we trust chapkit's own validation
    # and store user_option_values={} as a "use chapkit defaults" sentinel.
    chapkit_schema_user_options = {
        "n_lags": {
            "items": {"type": "integer"},
            "title": "N Lags",
            "type": "array",
        },
    }
    config = ModelTemplateConfigV2(
        name="chapkit_default_factory",
        version="1.0.0",
        required_covariates=["population"],
        allow_free_additional_continuous_covariates=False,
        user_options=chapkit_schema_user_options,
        meta_data=ModelTemplateMetaData(
            author="chap_temp",
            author_assessed_status="orange",
            description="chapkit model with default_factory field",
            display_name="Chapkit Default Factory",
        ),
    )
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(config)
        cm_id = session.add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={}),
            "default",
            uses_chapkit=True,
        )
        assert cm_id is not None
        with pytest.raises(ValueError, match="n_lags"):
            session.add_configured_model(
                template_id,
                ModelConfiguration(user_option_values={}),
                "no_chapkit",
                uses_chapkit=False,
            )


def test_yaml_update_preserves_uses_chapkit(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        template = session.session.get(ModelTemplateDB, template_id)
        template.uses_chapkit = True
        session.session.commit()

    with SessionWrapper(engine) as session:
        session.add_model_template_from_yaml_config(model_template_yaml_config)
        template = session.session.get(ModelTemplateDB, template_id)
        assert template.uses_chapkit is True


def test_new_template_version_is_added_as_new_row(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        model_template_yaml_config.version = "v1"
        v1_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        model_template_yaml_config.version = "v2"
        v2_id = session.add_model_template_from_yaml_config(model_template_yaml_config)

        assert v1_id != v2_id
        # The old version keeps its own row.
        assert session.get_model_template(v1_id).version == "v1"
        assert session.get_model_template(v2_id).version == "v2"


def test_first_template_version_is_live_before_it_has_a_configured_model(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)

        assert session.get_model_template(template_id).is_live is True


def test_new_template_version_stays_hidden_until_it_has_a_configured_model(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        model_template_yaml_config.version = "v1"
        v1_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        v1_configured_id = session.add_configured_model(v1_id, ModelConfiguration(user_option_values={}))

        model_template_yaml_config.version = "v2"
        v2_id = session.add_model_template_from_yaml_config(model_template_yaml_config)

        assert session.get_model_template(v1_id).is_live is True
        assert session.get_model_template(v2_id).is_live is False
        assert session.get_configured_model_by_name("test_model").id == v1_configured_id
        assert [model.id for model in session.get_configured_models()] == [v1_configured_id]

        v2_configured_id = session.add_configured_model(v2_id, ModelConfiguration(user_option_values={}))

        assert session.get_model_template(v1_id).is_live is False
        assert session.get_model_template(v2_id).is_live is True
        assert session.get_configured_model_by_name("test_model").id == v2_configured_id


def test_reseeding_changed_contents_under_the_same_version_keeps_stored_row(model_template_yaml_config, engine):
    """A version is write-once, so CHAP drops the edit."""
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        seeded_covariates = list(model_template_yaml_config.required_covariates)
        model_template_yaml_config.required_covariates = ["rainfall", "population"]
        model_template_yaml_config.meta_data.display_name = "Renamed model"

        assert session.add_model_template_from_yaml_config(model_template_yaml_config) == template_id

        template = session.get_model_template(template_id)
        assert template.required_covariates == seeded_covariates
        assert template.display_name != "Renamed model"


def test_drifted_template_content_fields_lists_the_changed_fields(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        stored = session.get_model_template(template_id)
        changed = stored.model_copy(
            update={"display_name": "Renamed model", "required_covariates": ["rainfall", "population"]}
        )

        assert drifted_template_content_fields(stored, changed) == ["display_name", "required_covariates"]
        assert drifted_template_content_fields(stored, stored.model_copy()) == []


def test_reseeding_older_template_version_makes_it_live_again(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        model_template_yaml_config.version = "v1"
        v1_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        v1_configured_id = session.add_configured_model(v1_id, ModelConfiguration(user_option_values={}))
        model_template_yaml_config.version = "v2"
        v2_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        session.add_configured_model(v2_id, ModelConfiguration(user_option_values={}))
        model_template_yaml_config.version = "v1"

        assert session.add_model_template_from_yaml_config(model_template_yaml_config) == v1_id
        assert session.get_model_template(v1_id).is_live is True
        assert session.get_model_template(v2_id).is_live is False
        assert session.get_configured_model_by_name("test_model").id == v1_configured_id


def test_new_template_version_gets_its_own_configured_model(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        model_template_yaml_config.version = "v1"
        v1_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        v1_configured_id = session.add_configured_model(v1_id, ModelConfiguration(user_option_values={}))
        model_template_yaml_config.version = "v2"
        v2_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        v2_configured_id = session.add_configured_model(v2_id, ModelConfiguration(user_option_values={}))

        assert v1_configured_id != v2_configured_id
        # CHAP offers only the live version.
        assert [model.id for model in session.get_configured_models()] == [v2_configured_id]
        assert session.get_configured_model_by_name("test_model").id == v2_configured_id
        # A pinned id still resolves the version that it points at.
        assert session.get_configured_model_by_id_or_name(v1_configured_id).id == v1_configured_id


def test_missing_configured_model_error_lists_only_live_names(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        model_template_yaml_config.version = "v1"
        v1_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        session.add_configured_model(v1_id, ModelConfiguration(user_option_values={}))
        session.add_configured_model(v1_id, ModelConfiguration(user_option_values={}), "legacy")
        model_template_yaml_config.version = "v2"
        v2_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        session.add_configured_model(v2_id, ModelConfiguration(user_option_values={}))

        with pytest.raises(ValueError, match="not found") as exc_info:
            session.get_configured_model_by_name("test_model:legacy")

        available = str(exc_info.value).split("Available names: ", 1)[1]
        assert available == "['test_model']"


def test_add_model_template_from_url_stores_source_digest(engine, model_template_yaml_config, monkeypatch):
    commit_sha = "0c41b1d9bd187521e62c58d581e6f5bd5127f7b5"
    fetched_urls = []

    def fetch_config(url):
        fetched_urls.append(url)
        return model_template_yaml_config

    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        fetch_config,
    )
    with SessionWrapper(engine) as session:
        template_id = add_model_template_from_url(
            f"https://github.com/example/test_model@{commit_sha}", session, version="v1"
        )
        assert session.get_model_template(template_id).source_digest == commit_sha
    assert fetched_urls == [f"https://github.com/example/test_model@{commit_sha}"]


def test_reseeding_a_moved_ref_keeps_the_originally_seeded_source(model_template_yaml_config, engine):
    """A branch ref such as @main can move, but the stored revision must not change."""
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config, source_digest="a" * 40)

        assert session.add_model_template_from_yaml_config(model_template_yaml_config, source_digest="b" * 40) == (
            template_id
        )
        # The row keeps its first revision.
        assert session.get_model_template(template_id).source_digest == "a" * 40


def test_changed_configuration_is_added_as_new_configured_model(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        # The shared fixture has no user options, and the schema is closed.
        model_template_yaml_config.user_options = {"n_lags": {"type": "integer"}}
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        first_id = session.add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={"n_lags": 3}, additional_continuous_covariates=["rainfall"]),
        )
        second_id = session.add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={"n_lags": 5}, additional_continuous_covariates=["rainfall"]),
        )

        assert first_id != second_id
        # The first configuration does not change.
        first = session.session.get(ConfiguredModelDB, first_id)
        assert first.additional_continuous_covariates == ["rainfall"]
        assert first.user_option_values == {"n_lags": 3}
        assert first.is_live is False
        assert [model.id for model in session.get_configured_models()] == [second_id]


def test_unchanged_configuration_reuses_configured_model(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        configuration = ModelConfiguration(user_option_values={}, additional_continuous_covariates=["rainfall"])
        first_id = session.add_configured_model(template_id, configuration)

        assert session.add_configured_model(template_id, configuration) == first_id
        assert [model.id for model in session.get_configured_models()] == [first_id]


@pytest.mark.parametrize("url", template_urls)
# @pytest.mark.slow
def test_add_model_template_from_url(engine, url):
    # url = 'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a'
    with SessionWrapper(engine) as session:
        template_id = add_model_template_from_url(url, session, version="test")
        configured_model_id = add_configured_model(
            template_id, ModelConfiguration(user_option_values={}), "default", session
        )
        external_model = session.get_configured_model_with_code(configured_model_id)
        assert isinstance(external_model, ExternalModel)


def test_add_model_template_from_url_name_override(engine, model_template_yaml_config, monkeypatch):
    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        lambda url: model_template_yaml_config,
    )
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: "a" * 40)
    with SessionWrapper(engine) as session:
        template_id = add_model_template_from_url(
            "https://github.com/example/test_model@main", session, version="test", name_override="my_distinct_name"
        )
        template = session.session.get(ModelTemplateDB, template_id)
        assert template.name == "my_distinct_name"


def test_add_model_template_from_url_requires_a_resolvable_source_digest(engine, monkeypatch):
    """CHAP does not store a git template if it cannot find the revision."""
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: None)
    with SessionWrapper(engine) as session, pytest.raises(ValueError, match="immutable source digest"):
        add_model_template_from_url("https://github.com/example/test_model@main", session, version="test")


def test_add_model_template_from_url_skips_github_when_version_exists(engine, model_template_yaml_config, monkeypatch):
    fetched_urls = []

    def fetch_config(url):
        fetched_urls.append(url)
        config = model_template_yaml_config.model_copy(deep=True)
        config.source_url = url
        return config

    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        fetch_config,
    )
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: "a" * 40)
    with SessionWrapper(engine) as session:
        first_id = add_model_template_from_url(
            "https://github.com/example/test_model@main",
            session,
            version="v1",
            name_override="test_model",
        )

        def fail_resolve(url):
            raise AssertionError("existing version must not resolve a commit")

        def fail_fetch(url):
            raise AssertionError("existing version must not fetch github")

        monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", fail_resolve)
        monkeypatch.setattr(
            "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
            fail_fetch,
        )
        second_id = add_model_template_from_url(
            "https://github.com/example/test_model@main",
            session,
            version="v1",
            name_override="test_model",
        )

        assert second_id == first_id
    assert fetched_urls == ["https://github.com/example/test_model@" + "a" * 40]


def test_add_model_template_from_url_skips_github_when_version_exists_without_name_override(
    engine, model_template_yaml_config, monkeypatch
):
    def fetch_config(url):
        config = model_template_yaml_config.model_copy(deep=True)
        config.source_url = url
        return config

    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        fetch_config,
    )
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: "a" * 40)
    with SessionWrapper(engine) as session:
        first_id = add_model_template_from_url("https://github.com/example/test_model@main", session, version="v1")

        def fail_resolve(url):
            raise AssertionError("existing version must not resolve a commit")

        def fail_fetch(url):
            raise AssertionError("existing version must not fetch github")

        monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", fail_resolve)
        monkeypatch.setattr(
            "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
            fail_fetch,
        )
        second_id = add_model_template_from_url("https://github.com/example/test_model@main", session, version="v1")

        assert second_id == first_id


def _two_git_model_config_dir(tmp_path):
    config_dir = tmp_path / "configured_models"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "- url: https://github.com/example/broken_model\n"
        "  name: broken_model\n"
        "  versions:\n"
        '    nightly_build: "@main"\n'
        "- url: https://github.com/example/ok_model\n"
        "  name: ok_model\n"
        "  versions:\n"
        '    v1: "@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
    )
    return config_dir


def _seeded_template_names(engine):
    with Session(engine) as session:
        return {template.name for template in session.exec(select(ModelTemplateDB)).all()}


def test_seed_skips_git_model_when_source_digest_cannot_be_resolved(
    engine, tmp_path, model_template_yaml_config, monkeypatch
):
    monkeypatch.setattr(
        "chap_core.database.model_template_seed.resolve_commit_sha",
        lambda url: None if "broken_model" in url else "a" * 40,
    )
    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        lambda url: model_template_yaml_config.model_copy(deep=True),
    )
    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, directory=_two_git_model_config_dir(tmp_path))

    names = _seeded_template_names(engine)
    assert "broken_model" not in names
    assert "ok_model" in names
    assert "naive_model" in names


def test_seed_skips_git_model_when_github_fetch_fails(engine, tmp_path, model_template_yaml_config, monkeypatch):
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: "a" * 40)

    def fetch_config(url):
        if "broken_model" in url:
            raise AssertionError("Error fetching MLProject file")
        return model_template_yaml_config.model_copy(deep=True)

    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        fetch_config,
    )
    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, directory=_two_git_model_config_dir(tmp_path))

    names = _seeded_template_names(engine)
    assert "broken_model" not in names
    assert "ok_model" in names
    assert "naive_model" in names


def _two_chapkit_model_config_dir(tmp_path):
    config_dir = tmp_path / "configured_models"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "- url: http://broken-chapkit:8000\n"
        "  uses_chapkit: true\n"
        "  versions:\n"
        '    v1: "/v1"\n'
        "- url: http://ok-chapkit:8000\n"
        "  uses_chapkit: true\n"
        "  versions:\n"
        '    v1: "/v1"\n'
    )
    return config_dir


def test_seed_skips_chapkit_model_when_version_is_missing(engine, tmp_path, model_template_yaml_config, monkeypatch):
    class FakeChapkitTemplate:
        def __init__(self, url):
            self.url = url

        def wait_for_healthy(self, timeout=30):
            return None

        def get_model_template_config(self):
            config = model_template_yaml_config.model_copy(deep=True)
            if "broken" in self.url:
                config.name = "broken_chapkit"
                config.version = None
                return config
            config.name = "ok_chapkit"
            return config

    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalChapkitModelTemplate",
        FakeChapkitTemplate,
    )
    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, directory=_two_chapkit_model_config_dir(tmp_path))

    names = _seeded_template_names(engine)
    assert "broken_chapkit" not in names
    assert "ok_chapkit" in names
    assert "naive_model" in names


def test_seed_raises_database_error_instead_of_hiding_model(engine, tmp_path, model_template_yaml_config, monkeypatch):
    monkeypatch.setattr("chap_core.database.model_template_seed.resolve_commit_sha", lambda url: "a" * 40)
    monkeypatch.setattr(
        "chap_core.database.model_template_seed.ExternalModelTemplate.fetch_config_from_github_url",
        lambda url: model_template_yaml_config.model_copy(deep=True),
    )
    # A legacy unique name constraint, as on a deployment where the (name, version) swap has not run.
    with engine.connect() as connection:
        connection.execute(text("CREATE UNIQUE INDEX legacy_name_key ON modeltemplatedb (name)"))
        connection.commit()
    with Session(engine) as session:
        session.add(ModelTemplateDB(name="broken_model", version="old"))
        session.commit()
        # broken_model's insert violates the legacy constraint mid-commit. A schema
        # error must fail startup rather than silently omit that model.
        with pytest.raises(IntegrityError):
            seed_configured_models_from_config_dir(session, directory=_two_git_model_config_dir(tmp_path))

    names = _seeded_template_names(engine)
    assert "ok_model" not in names
    assert "naive_model" not in names


def test_seed_configured_models(engine):
    # make sure is clean
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    # seed with models
    with Session(engine) as session:
        # ensure db doesnt contain any models
        configured_models = session.exec(select(ConfiguredModelDB)).all()
        assert not configured_models
        # seed with models
        seed_configured_models_from_config_dir(session, skip_chapkit_models=True)
        # seed again to check that repeated inserts are handled nicely
        seed_configured_models_from_config_dir(session, skip_chapkit_models=True)
    # test that models have been added
    with Session(engine) as session:
        configured_models = session.exec(select(ConfiguredModelDB).join(ConfiguredModelDB.model_template)).all()
        logger.info(f"A total of {len(configured_models)} configured models have been added to the db:")
        for m in configured_models:
            logger.info(f"--> {m}")
        assert len(configured_models) > 1
        model_names = [m.name for m in configured_models]
        assert "naive_model" in model_names


def test_seed_datasets_to_db(engine):
    with SessionWrapper(engine) as session:
        seed_example_datasets(session)


@pytest.fixture
def configured_model_fixture(engine, model_template_yaml_config):
    """Seed a model template + configured model and return (engine, configured_model_id, configured_model_name)."""
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        cm_id = session.add_configured_model(template_id, ModelConfiguration(user_option_values={}))
    with Session(engine) as s:
        cm = s.get(ConfiguredModelDB, cm_id)
        assert cm is not None
        name = cm.name
    return engine, cm_id, name


def test_configured_model_display_name(engine, model_template_yaml_config):
    with SessionWrapper(engine) as session:
        template_id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        default_id = session.add_configured_model(template_id, ModelConfiguration(user_option_values={}))
        named_id = session.add_configured_model(template_id, ModelConfiguration(user_option_values={}), "detail_view")

    template_display_name = model_template_yaml_config.meta_data.display_name
    with Session(engine) as s:
        default = s.get(ConfiguredModelDB, default_id)
        named = s.get(ConfiguredModelDB, named_id)
        assert default is not None and named is not None
        assert default.display_name == template_display_name
        assert named.display_name == f"{template_display_name} [Detail view]"


def test_resolve_configured_model_by_int_id(configured_model_fixture):
    engine, cm_id, expected_name = configured_model_fixture
    with SessionWrapper(engine) as session:
        result = session.get_configured_model_by_id_or_name(cm_id)
        assert result.id == cm_id
        assert result.name == expected_name


def test_resolve_configured_model_by_string_name(configured_model_fixture):
    engine, cm_id, expected_name = configured_model_fixture
    with SessionWrapper(engine) as session:
        result = session.get_configured_model_by_id_or_name(expected_name)
        assert result.id == cm_id
        assert result.name == expected_name


def test_resolve_configured_model_by_nonexistent_int_raises(configured_model_fixture):
    engine, _, _ = configured_model_fixture
    with SessionWrapper(engine) as session:
        with pytest.raises(ValueError, match="not found"):
            session.get_configured_model_by_id_or_name(99999)


def test_resolve_configured_model_by_nonexistent_name_raises(configured_model_fixture):
    engine, _, _ = configured_model_fixture
    with SessionWrapper(engine) as session:
        with pytest.raises(ValueError, match="not found"):
            session.get_configured_model_by_id_or_name("no_such_model")


def test_model_configuration_rejects_flat_yaml_keys():
    """Flat YAMLs like ``n_lag_periods: 5`` at the top level used to be silently dropped
    because pydantic ignored unknown fields. They must now raise a validation error so
    users learn to wrap parameters under ``user_option_values``."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelConfiguration.model_validate({"n_lag_periods": 5})


def test_model_configuration_accepts_nested_format():
    config = ModelConfiguration.model_validate(
        {
            "user_option_values": {"n_lag_periods": 5},
            "additional_continuous_covariates": ["rainfall"],
        }
    )
    assert config.user_option_values == {"n_lag_periods": 5}
    assert config.additional_continuous_covariates == ["rainfall"]
