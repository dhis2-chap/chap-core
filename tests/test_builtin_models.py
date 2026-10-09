import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSet, DataSetCreateInfo
from chap_core.database.model_template_seed import seed_builtin_models
from chap_core.database.model_templates_and_config_tables import ModelTemplateDB, ModelTemplateRole
from chap_core.database.tables import Backtest
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.file_io.example_data_set import datasets
from chap_core.models.builtin import (
    BuiltinModelTemplate,
    builtin_model,
    get_builtin_model,
    get_builtin_models,
)
from chap_core.models.model_template import ModelTemplate
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest, run_prediction
from chap_core.testing.estimators import sanity_check_estimator

BUILTIN_NAMES = sorted(get_builtin_models())


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def engine_with_dataset(engine, weekly_full_data):
    with SessionWrapper(engine) as session:
        DataSetManager(session.session).save_dataset(DataSetCreateInfo(name="full_data"), weekly_full_data, None)
        seed_builtin_models(session)
    return engine


def _global_median():
    return ModelTemplate.from_directory_or_github_url("builtin:global_median").get_model()()


def test_registering_a_name_twice_fails():
    with pytest.raises(ValueError, match="global_median"):
        builtin_model()(get_builtin_model("global_median"))


def test_unknown_builtin_url_fails():
    with pytest.raises(ValueError, match="no_such_model"):
        ModelTemplate.from_directory_or_github_url("builtin:no_such_model")


def test_builtin_url_resolves_the_registered_version_only():
    assert ModelTemplate.from_directory_or_github_url("builtin:global_median@1").name == "global_median"
    with pytest.raises(ValueError, match="version '0' cannot be run"):
        ModelTemplate.from_directory_or_github_url("builtin:global_median@0")


def test_global_median_samples_observations_of_the_same_location(health_population_data):
    train, test_generator = train_test_generator(health_population_data, prediction_length=3, n_test_sets=1)
    historic, future, _ = next(test_generator)

    forecasts = _global_median().train(train).predict(historic, future)
    repeated = _global_median().train(train).predict(historic, future)

    assert set(forecasts.keys()) == set(future.keys())
    for location, samples in forecasts.items():
        assert np.array_equal(samples.samples, repeated[location].samples)
        assert samples.samples.shape == (3, 100)
        observed = train[location].disease_cases
        assert set(samples.samples.ravel()) <= set(observed[np.isfinite(observed)])


def test_global_median_is_seeded_as_baseline(engine):
    with SessionWrapper(engine) as session:
        seed_builtin_models(session)
        seed_builtin_models(session)
        template = session.session.exec(select(ModelTemplateDB).where(ModelTemplateDB.name == "global_median")).one()
        assert template.role == ModelTemplateRole.baseline
        assert template.source_url == "builtin:global_median@1"
        assert session.get_configured_model_by_name("global_median").model_template_id == template.id


def test_unregistered_builtin_template_is_archived(engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_or_update_model_template(
            ModelTemplateDB(name="retired_builtin", version="1", source_url="builtin:retired_builtin")
        )
        seed_builtin_models(session)
        assert session.get_model_template(template_id).archived


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_model_passes_sanity_check(name):
    with ModelTemplate.from_directory_or_github_url(f"builtin:{name}") as template:
        assert isinstance(template, BuiltinModelTemplate)
        sanity_check_estimator(template.get_model())


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_model_runs_in_extended_predictor(name, weekly_full_data):
    model = ModelTemplate.from_directory_or_github_url(f"builtin:{name}").get_model()()
    model.model_information.max_prediction_periods = 1
    train, test_generator = train_test_generator(weekly_full_data, prediction_length=3, n_test_sets=1)
    historic, future, _ = next(test_generator)

    predictor = ExtendedPredictor(model, 3).train(train)
    forecasts = predictor.predict(historic, future)

    assert set(forecasts.keys()) == set(future.keys())
    assert all(len(samples) == 3 for samples in forecasts.values())


@pytest.mark.parametrize("name", BUILTIN_NAMES)
@pytest.mark.parametrize("dry_run", [False, True])
def test_builtin_model_runs_in_chap_eval(name, dry_run, tmp_path):
    csv_path = tmp_path / "data.csv"
    datasets["hydromet_5_filtered"].load().to_csv(csv_path)
    output_file = tmp_path / "evaluation.nc"

    eval_cmd(
        model_name=f"builtin:{name}",
        dataset_csv=csv_path,
        output_file=output_file,
        backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
        run_config=RunConfig(),
        dry_run=dry_run,
    )

    assert output_file.exists() != dry_run


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_model_runs_backtest_and_prediction(name, engine_with_dataset):
    with SessionWrapper(engine_with_dataset) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = run_backtest(
            BacktestCreate(name="builtin", dataset_id=dataset_id, model_id=name),
            n_periods=3,
            n_splits=2,
            stride=1,
            session=session,
        )
        prediction_id = run_prediction(name, dataset_id, 3, name="builtin", session=session)

    with Session(engine_with_dataset) as session:
        backtest = session.get(Backtest, backtest_id)
        assert backtest is not None and backtest.forecasts
        assert prediction_id is not None


def test_chap_report_rejects_builtin_models(tmp_path):
    from chap_core.cli_endpoints.report import report

    csv_path = tmp_path / "data.csv"
    datasets["hydromet_5_filtered"].load().to_csv(csv_path)

    with pytest.raises(ValueError, match="built-in"):
        report(model_name="builtin:global_median", dataset_csv=csv_path, out_file=tmp_path / "report.pdf")
