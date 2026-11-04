import numpy as np
import pytest
from sqlmodel import Session, select

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSet, DataSetCreateInfo
from chap_core.database.model_template_seed import seed_builtin_models
from chap_core.database.model_templates_and_config_tables import ModelTemplateDB, ModelTemplateRole
from chap_core.database.tables import Backtest
from chap_core.datatypes import FullData
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.file_io.example_data_set import datasets
from chap_core.models.builtin import (
    BaselineNotApplicable,
    BuiltinModelTemplate,
    builtin_model,
    get_builtin_model,
    get_builtin_models,
)
from chap_core.models.builtin.empirical import period_of_year, quantile_samples
from chap_core.models.builtin.persistence import SeasonalShifts
from chap_core.models.model_template import ModelTemplate
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest, run_prediction
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet
from chap_core.testing.estimators import sanity_check_estimator

BUILTIN_NAMES = sorted(get_builtin_models())


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
    version = get_builtin_model("global_median").spec.version
    assert ModelTemplate.from_directory_or_github_url(f"builtin:global_median@{version}").name == "global_median"
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


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_model_is_seeded_as_baseline(name, engine):
    with SessionWrapper(engine) as session:
        seed_builtin_models(session)
        seed_builtin_models(session)
        template = session.session.exec(select(ModelTemplateDB).where(ModelTemplateDB.name == name)).one()
        assert template.role == ModelTemplateRole.baseline
        assert template.source_url == f"builtin:{name}@{get_builtin_model(name).spec.version}"
        assert session.get_configured_model_by_name(name).model_template_id == template.id


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


def _other_locations_median(train, location):
    return np.median([np.nanmedian(train[other].disease_cases) for other in train.keys() if other != location])


def test_global_median_uses_other_locations_for_an_unseen_location(weekly_split):
    train, historic, future = weekly_split
    others = [location for location in train.keys() if location != "boaco"]

    samples = _forecast("global_median", train.filter_locations(others), historic, future)["boaco"].samples

    assert np.all(samples[:, 49] == _other_locations_median(train, "boaco"))
    assert np.all(np.diff(samples, axis=1) >= 0)


def test_global_median_uses_other_locations_for_a_location_without_observations(weekly_split):
    train, historic, future = weekly_split
    df = train.to_pandas()
    df.loc[df.location == "boaco", "disease_cases"] = np.nan
    unobserved = SpatioTemporalDataSet.from_pandas(df, FullData)

    samples = _forecast("global_median", unobserved, historic, future)["boaco"].samples

    assert np.all(samples[:, 49] == _other_locations_median(train, "boaco"))


def test_global_median_is_not_applicable_without_any_observations(weekly_split):
    train, historic, future = weekly_split
    df = train.to_pandas()
    df["disease_cases"] = np.nan
    with pytest.raises(BaselineNotApplicable, match="any location"):
        _forecast("global_median", SpatioTemporalDataSet.from_pandas(df, FullData), historic, future)


def test_seasonal_median_is_the_median_of_the_same_week(weekly_split):
    train, historic, future = weekly_split
    forecasts = _forecast("seasonal_median", train, historic, future)
    for location, samples in forecasts.items():
        cases = train[location].disease_cases
        for week, median in zip(future[location].time_period.week, samples.samples[:, 49], strict=True):
            same_week = cases[train[location].time_period.week == week]
            assert median == np.nanmedian(same_week)


def test_seasonal_median_uses_the_global_median_for_an_unobserved_period_of_year(weekly_full_data):
    periods = weekly_full_data.period_range
    short_train = weekly_full_data.restrict_time_period(slice(periods[0], periods[9]))
    future = weekly_full_data.restrict_time_period(slice(periods[20], periods[22]))

    seasonal = _forecast("seasonal_median", short_train, short_train, future)
    global_median = _forecast("global_median", short_train, short_train, future)

    for location, samples in seasonal.items():
        np.testing.assert_array_equal(samples.samples, global_median[location].samples)


def test_seasonal_median_uses_other_locations_for_an_unseen_location(weekly_split):
    train, historic, future = weekly_split
    others = [location for location in train.keys() if location != "boaco"]

    samples = _forecast("seasonal_median", train.filter_locations(others), historic, future)["boaco"].samples

    assert np.all(samples[:, 49] == _other_locations_median(train, "boaco"))


def test_persistence_scales_the_last_historic_value(weekly_split):
    train, historic, future = weekly_split
    last_value = historic["boaco"].disease_cases[-1]
    df = historic.to_pandas()
    df.loc[(df.location == "boaco") & (df.time_period == df.time_period.max()), "disease_cases"] = 2 * last_value
    doubled = SpatioTemporalDataSet.from_pandas(df, FullData)

    median = _forecast("persistence", train, historic, future)["boaco"].samples[:, 49]
    doubled_median = _forecast("persistence", train, doubled, future)["boaco"].samples[:, 49]

    np.testing.assert_allclose(doubled_median + 1, (median + 1) * (2 * last_value + 1) / (last_value + 1))


def test_persistence_forecasts_with_spread_beyond_a_short_history(weekly_full_data):
    periods = weekly_full_data.period_range
    short_train = weekly_full_data.restrict_time_period(slice(periods[0], periods[9]))
    future = weekly_full_data.restrict_time_period(slice(periods[10], periods[21]))

    forecasts = _forecast("persistence", short_train, short_train, future)

    assert set(forecasts.keys()) == set(future.keys())
    for samples in forecasts.values():
        assert np.all(np.isfinite(samples.samples))
        # Ten periods have one pair nine periods apart, so horizon 9 and later need all locations' ratios.
        assert np.all(np.ptp(samples.samples[8:], axis=1) > 0)


def test_persistence_forecasts_with_spread_more_than_a_year_after_the_last_value(weekly_split):
    train, historic, future = weekly_split
    df = historic.to_pandas()
    gap = df.time_period.isin(df.time_period.unique()[-60:])
    df.loc[(df.location == "boaco") & gap, "disease_cases"] = np.nan

    samples = _forecast("persistence", train, SpatioTemporalDataSet.from_pandas(df, FullData), future)["boaco"].samples

    assert np.all(np.ptp(samples, axis=1) > 0)


def test_persistence_uses_the_ratios_of_all_locations_for_an_unseen_location(weekly_split):
    train, historic, future = weekly_split
    others = train.filter_locations([location for location in train.keys() if location != "boaco"])
    pooled = SeasonalShifts.pooled(
        SeasonalShifts.from_cases(np.asarray(data.disease_cases, dtype=float), period_of_year(data.time_period), 52)
        for data in others.values()
    )

    samples = _forecast("persistence", others, historic, future)["boaco"].samples

    last_value = historic["boaco"].disease_cases[-1]
    first_week = period_of_year(future["boaco"].time_period)[0]
    expected = np.maximum((last_value + 1) * quantile_samples(pooled.ratios(first_week, 1)) - 1, 0)
    np.testing.assert_allclose(samples[0], expected)


def test_persistence_forecasts_the_last_value_without_any_ratios(weekly_full_data):
    periods = weekly_full_data.period_range
    single_period = weekly_full_data.restrict_time_period(slice(periods[0], periods[0]))
    future = weekly_full_data.restrict_time_period(slice(periods[1], periods[2]))

    forecasts = _forecast("persistence", single_period, single_period, future)

    for location, samples in forecasts.items():
        assert np.all(samples.samples == single_period[location].disease_cases[-1])


def test_persistence_is_not_applicable_without_a_historic_value(weekly_split):
    train, historic, future = weekly_split
    df = historic.to_pandas()
    df.loc[df.location == "boaco", "disease_cases"] = np.nan
    with pytest.raises(BaselineNotApplicable, match="boaco"):
        _forecast("persistence", train, SpatioTemporalDataSet.from_pandas(df, FullData), future)


SHIFTS = SeasonalShifts({1: {2: np.array([1.0, 3.0]), 3: np.array([2.0]), 5: np.array([4.0]), 8: np.array([6.0])}})


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (3, [2.0]),
        (4, [2.0, 4.0]),
        (6, [4.0, 6.0]),
        (1, [1.0, 3.0, 2.0, 4.0, 6.0]),
        (9, [1.0, 3.0, 2.0, 4.0, 6.0]),
    ],
    ids=["observed", "one-missing", "several-missing", "nothing-earlier", "nothing-later"],
)
def test_seasonal_shifts_fall_back_to_the_nearest_periods_of_year(season, expected):
    np.testing.assert_array_equal(SHIFTS.ratios(season, horizon=1), expected)


def test_seasonal_shifts_have_no_ratios_for_an_unseen_horizon():
    assert SHIFTS.ratios(3, horizon=2) is None
