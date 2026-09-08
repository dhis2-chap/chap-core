"""MSTL + ARIMA provider.

Decomposes each covariate into seasonal, trend and remainder components with
MSTL (multiple seasonal-trend decomposition using LOESS), forecasts the
deseasonalised series with an ARIMA, and adds the seasonal component back by
continuing it forward. This follows nixtla's MSTL model, including automatic
ARIMA order selection.

Orders are chosen per series the way Hyndman-Khandakar's auto.arima does: the
number of differences from a KPSS stationarity test, then a search over (p, q)
scored by AICc. Pass ``params={"order": (p, d, q)}`` to skip the search and fit
a fixed order, which is much faster.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from chap_core.assessment.weather_providers import weather_provider
from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase

#: Largest AR and MA orders considered by the automatic search.
MAX_P = 3
MAX_Q = 3

#: Largest number of differences the KPSS test may ask for.
MAX_D = 2

#: Fallback when no candidate order can be fitted at all.
FALLBACK_ARIMA_ORDER = (0, 0, 0)

#: Candidate models whose characteristic roots sit this close to the unit circle
#: are rejected. Their forecasts oscillate explosively - on Vietnam admin1 data a
#: handful of such models produced rainfall forecasts of +-5000 against a
#: historical range of 1-18, and 77% of all forecast error came from the worst 1%
#: of series. auto.arima applies the same guard.
MIN_ROOT_MODULUS = 1.001

#: History must be longer than this many full cycles. MSTL discards any period
#: that is not shorter than half the series, so exactly two cycles yields no
#: seasonal component at all.
MIN_SEASONAL_CYCLES = 2


def _seasonal_period(period_range) -> int:
    from chap_core.time_period import Month, Week

    first = period_range[0]
    if isinstance(first, Month):
        return 12
    if isinstance(first, Week):
        return 52
    raise ValueError(f"mstl_arima does not know the seasonal period for {type(first).__name__} data.")


def _n_differences(y: np.ndarray, max_d: int = MAX_D) -> int:
    """Number of differences needed for stationarity, per a KPSS test.

    KPSS tests the null that the series is stationary, so a p-value above the
    5% level means we stop differencing.
    """
    from statsmodels.tsa.stattools import kpss

    series = y
    for d in range(max_d):
        if len(series) < 10 or np.allclose(series, series[0]):
            return d
        try:
            p_value = kpss(series, regression="c", nlags="auto")[1]
        except (ValueError, OverflowError):
            return d
        if p_value > 0.05:
            return d
        series = np.diff(series)
    return max_d


def _is_well_conditioned(fitted) -> bool:
    """Reject models with characteristic roots on or near the unit circle.

    Stationarity only requires the roots to lie outside the unit circle; one
    sitting arbitrarily close to it satisfies the constraint while forecasting an
    explosive oscillation.
    """
    for roots in (fitted.arroots, fitted.maroots):
        if len(roots) and np.min(np.abs(roots)) < MIN_ROOT_MODULUS:
            return False
    return True


def _select_order(y: np.ndarray) -> tuple[int, int, int]:
    """Pick (p, d, q) by AICc over a bounded grid, as auto.arima does."""
    from statsmodels.tsa.arima.model import ARIMA

    d = _n_differences(y)
    best_order = None
    best_score = np.inf
    for p in range(MAX_P + 1):
        for q in range(MAX_Q + 1):
            try:
                fitted = ARIMA(y, order=(p, d, q)).fit()
            except Exception:
                continue
            if not _is_well_conditioned(fitted):
                continue
            score = fitted.aicc
            if np.isfinite(score) and score < best_score:
                best_score, best_order = score, (p, d, q)
    return best_order or FALLBACK_ARIMA_ORDER


def _forecast_series(
    y: np.ndarray,
    n_periods: int,
    seasonal_period: int,
    order: tuple[int, int, int] | None,
) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import MSTL

    if len(y) <= MIN_SEASONAL_CYCLES * seasonal_period:
        # MSTL discards any period that is not shorter than half the series, so at
        # exactly two cycles it produces no seasonal component at all and raises an
        # UnboundLocalError from inside statsmodels. Reject that boundary here with
        # a message the caller can act on.
        raise ValueError(
            f"mstl_arima needs more than {MIN_SEASONAL_CYCLES} full seasonal cycles "
            f"(more than {MIN_SEASONAL_CYCLES * seasonal_period} periods) of history, got {len(y)}. "
            "Use the 'climatology' provider for shorter series."
        )

    with warnings.catch_warnings():
        # Short climate series routinely trip convergence and p-value-bound
        # warnings; one per location per covariate per split would drown the
        # evaluation log.
        warnings.simplefilter("ignore")
        decomposition = MSTL(y, periods=seasonal_period).fit()
        seasonal = np.asarray(decomposition.seasonal, dtype=float)
        if seasonal.ndim > 1:
            seasonal = seasonal.sum(axis=1)
        deseasonalised = y - seasonal
        chosen = order if order is not None else _select_order(deseasonalised)
        trend_forecast = np.asarray(ARIMA(deseasonalised, order=chosen).fit().forecast(n_periods), dtype=float)

    # Continue the seasonal component by averaging every complete cycle rather
    # than replaying the most recent one. MSTL's seasonal component varies over
    # time, so a single cycle is one year of noise, and it is the cycle at the
    # series edge where the LOESS fit is weakest. Averaging is what makes
    # climatology's pooled month-of-year mean hard to beat; taking only the last
    # cycle cost 10-27% MAE against it on Vietnam and Malawi admin data.
    n_cycles = len(seasonal) // seasonal_period
    cycles = seasonal[len(seasonal) - n_cycles * seasonal_period :].reshape(n_cycles, seasonal_period)
    # Row 0 starts a whole number of cycles back, so its phase relative to the
    # first forecast period is 0.
    future_seasonal = cycles.mean(axis=0)[np.arange(n_periods) % seasonal_period]
    return trend_forecast + future_seasonal


@weather_provider(
    "mstl_arima",
    "MSTL + ARIMA",
    "Seasonal-trend decomposition (LOESS) with an automatically ordered ARIMA on the "
    "deseasonalised series and the seasonal component continued forward. Needs more than "
    "two full seasonal cycles of history.",
)
class MstlArimaWeatherProvider(FutureWeatherProviderBase):
    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        # Imported lazily: temporal_dataclass imports api_types, which imports this
        # registry for the BacktestParams default.
        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

        params = params or {}
        order = params.get("order")
        order = tuple(order) if order is not None else None
        seasonal_period = params.get("seasonal_period") or _seasonal_period(historical_data.period_range)
        n_periods = len(period_range)

        historical_data = historical_data.remove_field("disease_cases")
        prediction_dict = {}
        for location, data in historical_data.items():
            fields = {}
            for field in dataclasses.fields(data):
                if field.name == "time_period":
                    continue
                y = getattr(data, field.name)
                if y.dtype.kind not in ("f", "i"):
                    # No seasonal signal to decompose; carry the last value forward.
                    fields[field.name] = y[-1:][np.zeros(n_periods, dtype=int)]
                    continue
                fields[field.name] = _forecast_series(np.asarray(y, dtype=float), n_periods, seasonal_period, order)
            prediction_dict[location] = data.__class__(period_range, **fields)
        return DataSet(prediction_dict)
