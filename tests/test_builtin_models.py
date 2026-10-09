import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSet, DataSetCreateInfo
from chap_core.database.model_template_seed import add_configured_model, seed_builtin_models
from chap_core.database.model_templates_and_config_tables import ModelConfiguration, ModelTemplateDB, ModelTemplateRole
from chap_core.database.tables import Backtest
from chap_core.datatypes import FullData
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
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet
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


def test_global_median_samples_span_observations_of_the_same_location(health_population_data):
    train, test_generator = train_test_generator(health_population_data, prediction_length=3, n_test_sets=1)
    historic, future, _ = next(test_generator)

    forecasts = _global_median().train(train).predict(historic, future)
    repeated = _global_median().train(train).predict(historic, future)

    assert set(forecasts.keys()) == set(future.keys())
    for location, samples in forecasts.items():
        assert np.array_equal(samples.samples, repeated[location].samples)
        assert samples.samples.shape == (3, 100)
        observed = train[location].disease_cases
        assert np.nanmin(observed) <= samples.samples.min() <= samples.samples.max() <= np.nanmax(observed)
        assert np.all(samples.samples[:, 49] == np.nanmedian(observed))


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


@pytest.fixture
def weekly_split(weekly_full_data):
    """The Nicaragua weekly data split into training data and four future weeks, 2022W41 to 2022W44."""
    train, test_generator = train_test_generator(weekly_full_data, prediction_length=4, n_test_sets=1)
    historic, future, _ = next(test_generator)
    return train, historic, future


def _forecast(name, train, historic, future):
    return (
        ModelTemplate.from_directory_or_github_url(f"builtin:{name}")
        .get_model()()
        .train(train)
        .predict(historic, future)
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("global_median", [[0.0, 18.0, 18.0, 196.303]] * 4),
        (
            "seasonal_median",
            [
                [2.2576, 25.5, 25.5, 150.7424],
                [2.4293, 27.0, 27.0, 161.798],
                [2.6869, 42.5, 42.5, 150.9141],
                [2.1717, 34.0, 34.0, 176.9949],
            ],
        ),
        (
            "persistence",
            [
                [56.0746, 109.9117, 109.9117, 217.5599],
                [50.8149, 99.0098, 99.0098, 246.4415],
                [38.1871, 115.699, 115.699, 784.4159],
                [16.2527, 126.6284, 126.6284, 1628.1958],
            ],
        ),
    ],
)
def test_baseline_fixed_output(name, expected, weekly_split):
    samples = _forecast(name, *weekly_split)["boaco"].samples
    assert samples.shape == (4, 100)
    # Lowest, the two middle and highest sample of each future week.
    np.testing.assert_allclose(samples[:, [0, 49, 50, 99]], expected, atol=1e-4)


@pytest.mark.parametrize("name", ["global_median", "seasonal_median", "persistence"])
def test_baseline_median_is_the_middle_sample(name, weekly_split):
    forecasts = _forecast(name, *weekly_split)
    for samples in forecasts.values():
        middle = samples.samples[:, 49]
        assert np.array_equal(samples.samples[:, 50], middle)
        # Both median definitions the metrics use give the point forecast exactly.
        assert np.array_equal(np.median(samples.samples, axis=1), middle)
        assert np.array_equal(pd.DataFrame(samples.samples.T).median().to_numpy(), middle)


def test_seasonal_median_is_the_median_of_the_same_week(weekly_split):
    train, historic, future = weekly_split
    forecasts = _forecast("seasonal_median", train, historic, future)
    for location, samples in forecasts.items():
        cases = train[location].disease_cases
        for week, median in zip(future[location].time_period.week, samples.samples[:, 49], strict=True):
            same_week = cases[train[location].time_period.week == week]
            assert median == np.nanmedian(same_week)


def test_seasonal_median_fails_for_an_unobserved_period_of_year(weekly_full_data):
    periods = weekly_full_data.period_range
    short_train = weekly_full_data.restrict_time_period(slice(periods[0], periods[9]))
    future = weekly_full_data.restrict_time_period(slice(periods[20], periods[22]))
    with pytest.raises(ValueError, match="needs every forecast period of the year observed"):
        _forecast("seasonal_median", short_train, short_train, future)


def test_persistence_scales_the_last_historic_value(weekly_split):
    train, historic, future = weekly_split
    last_value = historic["boaco"].disease_cases[-1]
    df = historic.to_pandas()
    df.loc[(df.location == "boaco") & (df.time_period == df.time_period.max()), "disease_cases"] = 2 * last_value
    doubled = SpatioTemporalDataSet.from_pandas(df, FullData)

    median = _forecast("persistence", train, historic, future)["boaco"].samples[:, 49]
    doubled_median = _forecast("persistence", train, doubled, future)["boaco"].samples[:, 49]

    np.testing.assert_allclose(doubled_median + 1, (median + 1) * (2 * last_value + 1) / (last_value + 1))


def test_persistence_fails_without_earlier_changes_to_learn_from(weekly_full_data):
    periods = weekly_full_data.period_range
    short_train = weekly_full_data.restrict_time_period(slice(periods[0], periods[9]))
    future = weekly_full_data.restrict_time_period(slice(periods[10], periods[11]))
    with pytest.raises(ValueError, match="Persistence needs one to estimate the spread"):
        _forecast("persistence", short_train, short_train, future)


@pytest.mark.parametrize("name", ["global_median", "seasonal_median", "persistence"])
def test_baselines_are_seeded_with_the_baseline_role(name, engine):
    with SessionWrapper(engine) as session:
        seed_builtin_models(session)
        template = session.session.exec(select(ModelTemplateDB).where(ModelTemplateDB.name == name)).one()
        assert template.role == ModelTemplateRole.baseline


def test_named_configuration_of_a_builtin_model_runs_the_builtin_model(engine, weekly_split):
    train, historic, future = weekly_split
    with SessionWrapper(engine) as session:
        seed_builtin_models(session)
        template = session.session.exec(select(ModelTemplateDB).where(ModelTemplateDB.name == "global_median")).one()
        configured_model_id = add_configured_model(
            template.id,
            ModelConfiguration(additional_continuous_covariates=[], user_option_values={}),
            "custom",
            session,
        )
        model = session.get_configured_model_with_code(configured_model_id)
        forecasts = model.train(train).predict(historic, future)
    assert set(forecasts.keys()) == set(future.keys())


def test_configuration_pinned_to_another_builtin_version_is_refused(engine):
    with SessionWrapper(engine) as session:
        template_id = session.add_or_update_model_template(
            ModelTemplateDB(name="global_median", version="0", source_url="builtin:global_median@0")
        )
        configured_model_id = add_configured_model(
            template_id,
            ModelConfiguration(additional_continuous_covariates=[], user_option_values={}),
            "default",
            session,
        )
        with pytest.raises(ValueError, match="version '0' cannot be run"):
            session.get_configured_model_with_code(configured_model_id)
