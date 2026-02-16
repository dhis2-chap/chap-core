"""Train/test splitting utilities for time series evaluation.

Provides functions for splitting spatio-temporal datasets into training and
test sets for model evaluation. The main entry point is ``train_test_generator``,
which implements an expanding window cross-validation strategy where the
training set grows with each successive split while the prediction window
slides forward.
"""

from collections.abc import Iterable, Iterator
from typing import Protocol

from chap_core.climate_predictor import FutureWeatherFetcher
from chap_core.datatypes import ClimateData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import TimePeriod
from chap_core.time_period.relationships import previous


class IsTimeDelta(Protocol):
    pass


def split_test_train_on_period(
    data_set: DataSet,
    split_points: Iterable[TimePeriod],
    future_length: IsTimeDelta | None = None,
    include_future_weather: bool = False,
    future_weather_class: type[ClimateData] = ClimateData,
):
    """Generate train/test splits at each split point.

    For each split point, produces a (train, test) tuple where training data
    ends just before the split point and test data starts at the split point.

    Parameters
    ----------
    data_set
        The full dataset to split.
    split_points
        Time periods at which to split (each becomes the start of a test set).
    future_length
        Optional time delta to limit test set length.
    include_future_weather
        If True, return (train, test, future_weather) tuples instead.
    future_weather_class
        Dataclass type for future weather data.

    Yields
    ------
    tuple
        (train, test) or (train, test, future_weather) for each split point.
    """
    func = train_test_split_with_weather if include_future_weather else train_test_split

    if include_future_weather:
        return (
            train_test_split_with_weather(data_set, period, future_length, future_weather_class)
            for period in split_points
        )
    return (func(data_set, period, future_length) for period in split_points)


def train_test_split(
    data_set: DataSet,
    prediction_start_period: TimePeriod,
    extension: IsTimeDelta | None = None,
    restrict_test=True,
):
    """Split a dataset into train and test sets at a single split point.

    Parameters
    ----------
    data_set
        The full dataset.
    prediction_start_period
        First period of the test set. Training data ends at the period
        immediately before this.
    extension
        Optional time delta to extend the test set end beyond
        ``prediction_start_period``.
    restrict_test
        If True, restrict the test set to the prediction window.
        If False, return the full dataset as the test set.

    Returns
    -------
    tuple[DataSet, DataSet]
        (train_data, test_data).
    """
    last_train_period = previous(prediction_start_period)
    train_data = data_set.restrict_time_period(slice(None, last_train_period))
    if extension is not None:
        end_period = prediction_start_period.extend_to(extension)
    else:
        end_period = None
    if restrict_test:
        test_data = data_set.restrict_time_period(slice(prediction_start_period, end_period))
    else:
        test_data = data_set
    return train_data, test_data


def train_test_generator(
    dataset: DataSet,
    prediction_length: int,
    n_test_sets: int = 1,
    stride: int = 1,
    future_weather_provider: FutureWeatherFetcher | None = None,
) -> tuple[DataSet, Iterator[tuple[DataSet, DataSet, DataSet]]]:
    """Generate expanding-window train/test splits for backtesting.

    Implements an expanding window cross-validation strategy. A fixed training
    set is returned (used to train the model once), along with an iterator of
    ``n_test_sets`` splits. Each split consists of:

    - **historic_data**: all data up to the split point (expands each split)
    - **masked_future_data**: future covariates *without* disease_cases
    - **future_data**: full future data including disease_cases (ground truth)

    The split indices are computed from the end of the dataset working
    backwards::

        split_idx = -(prediction_length + (n_test_sets - 1) * stride + 1)

    Example with a dataset of 20 periods, prediction_length=3, n_test_sets=2,
    stride=1::

        split_idx = -(3 + (2 - 1) * 1 + 1) = -5  -> index 15

        Split 0: historic = periods[..15], future = periods[16..18]
        Split 1: historic = periods[..16], future = periods[17..19]

        Train set = periods[..15]  (same as split 0 historic)

    Parameters
    ----------
    dataset
        The full dataset.
    prediction_length
        Number of periods to predict in each test window.
    n_test_sets
        Number of test splits to generate.
    stride
        Number of periods to advance between successive splits.
    future_weather_provider
        Optional callable that provides future weather data (with
        disease_cases masked) for each test split.

    Returns
    -------
    tuple[DataSet, Iterable[tuple[DataSet, DataSet, DataSet]]]
        The training set and an iterator of
        (historic_data, masked_future_data, future_data) tuples.
    """
    split_idx = -(prediction_length + (n_test_sets - 1) * stride + 1)
    train_set = dataset.restrict_time_period(slice(None, dataset.period_range[split_idx]))
    historic_data = [
        dataset.restrict_time_period(slice(None, dataset.period_range[split_idx + i * stride]))
        for i in range(n_test_sets)
    ]
    future_data = [
        dataset.restrict_time_period(
            slice(
                dataset.period_range[split_idx + i * stride + 1],
                dataset.period_range[split_idx + i * stride + prediction_length],
            )
        )
        for i in range(n_test_sets)
    ]
    if future_weather_provider is not None:
        masked_future_data = [
            future_weather_provider(hd).get_future_weather(fd.period_range)  # type: ignore[operator]
            for (hd, fd) in zip(historic_data, future_data)
        ]
    else:
        masked_future_data = [dataset.remove_field("disease_cases") for dataset in future_data]
    train_set.metadata = dataset.metadata.model_copy()
    train_set.metadata.name += "_train_set"
    return train_set, zip(historic_data, masked_future_data, future_data)


def train_test_split_with_weather(
    data_set: DataSet,
    prediction_start_period: TimePeriod,
    extension: IsTimeDelta | None = None,
    future_weather_class: type[ClimateData] = ClimateData,
):
    train_set, test_set = train_test_split(data_set, prediction_start_period, extension)
    future_weather = test_set.remove_field("disease_cases")
    train_periods = {str(period) for data in train_set.data() for period in data.data().time_period}
    future_periods = {str(period) for data in future_weather.data() for period in data.data().time_period}
    assert train_periods & future_periods == set(), (
        f"Train and future weather data overlap: {train_periods & future_periods}"
    )
    return train_set, test_set, future_weather


def get_split_points_for_data_set(data_set: DataSet, max_splits: int, start_offset=1) -> list[TimePeriod]:
    """Compute evenly-spaced split points for a dataset.

    Uses the time periods from the first location (assumes all locations share
    the same time range).

    Parameters
    ----------
    data_set
        The dataset to compute split points for.
    max_splits
        Maximum number of split points to return.
    start_offset
        Number of initial periods to skip before the first possible split.

    Returns
    -------
    list[TimePeriod]
        Up to ``max_splits`` evenly-spaced time periods.
    """
    periods = (
        next(iter(data_set.data())).data().time_period
    )  # Uses the time for the first location, assumes it to be the same for all!
    return get_split_points_for_period_range(max_splits, periods, start_offset)


def get_split_points_for_period_range(max_splits: int, periods, start_offset: int) -> list[TimePeriod]:
    """Compute evenly-spaced split points from a period range.

    Divides the available periods (after ``start_offset``) into
    ``max_splits + 1`` equal segments and returns the boundary points.

    Parameters
    ----------
    max_splits
        Maximum number of split points to return.
    periods
        Sequence of time periods to select from.
    start_offset
        Number of initial periods to skip.

    Returns
    -------
    list[TimePeriod]
        Up to ``max_splits`` evenly-spaced time periods.
    """
    delta = (len(periods) - 1 - start_offset) // (max_splits + 1)
    return list(periods)[start_offset + delta :: delta][:max_splits]
