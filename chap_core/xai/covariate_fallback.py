import calendar
import logging
import re
from contextlib import suppress
from datetime import date
from typing import Any

import pandas as pd

from chap_core.time_period.date_util_wrapper import Day, Month, TimePeriod, Week, clean_timestring

logger = logging.getLogger(__name__)

_MONTH_HEAD_RE = re.compile(r"^(\d{4})[-/](\d{1,2})(?:[-/T\s]|$)")
_DAY_HEAD_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def year_month_from_any(raw: Any) -> tuple[int, int] | None:
    """Extract (year, month) from any period-like value, or return None if not parseable.

    Handles pd.Period, pd.Timestamp, date objects, numpy scalars, integer YYYYMM values,
    string representations (e.g. "202405", "2024-05", "2024-05-01"), and falls back to
    pd.to_datetime for anything else.
    """
    if raw is None:
        return None
    if isinstance(raw, pd.Period):
        return (int(raw.year), int(raw.month))
    if isinstance(raw, pd.Timestamp):
        if pd.isna(raw):
            return None
        return (int(raw.year), int(raw.month))
    if isinstance(raw, date):
        return (raw.year, raw.month)
    if hasattr(raw, "item"):
        with suppress(Exception):
            raw = raw.item()
    if (
        isinstance(raw, (int, float))
        and not (isinstance(raw, float) and pd.isna(raw))
        and float(raw).is_integer()
        and 190001 <= int(raw) <= 999912
    ):
        v = int(raw)
        s = str(v)
        if len(s) == 6:
            return (int(s[:4]), int(s[4:6]))
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "nat", "none", "<nat>"):
        return None
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit() and len(head) == 6:
            return (int(head[:4]), int(head[4:6]))
    if len(s) == 6 and s.isdigit():
        return (int(s[:4]), int(s[4:6]))
    m = _MONTH_HEAD_RE.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = _DAY_HEAD_RE.match(s)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)))
    head = s.split()[0] if " " in s else s
    if len(head) == 8 and head.isdigit():
        try:
            tp = TimePeriod.from_id(head)
            if isinstance(tp, Month):
                return (tp.year, tp.month)
            if isinstance(tp, Day):
                return (tp.year, tp.month)
        except Exception:
            pass
    try:
        tp = TimePeriod.parse(head)
        if isinstance(tp, Month):
            return (tp.year, tp.month)
        if isinstance(tp, Day):
            return (tp.year, tp.month)
    except Exception:
        pass
    try:
        ts = pd.to_datetime(raw, errors="coerce")
        if isinstance(ts, pd.Timestamp) and pd.notna(ts):
            return (int(ts.year), int(ts.month))
    except Exception:
        pass
    return None


def covariate_period_candidates(forecast_period: str) -> list[str]:
    fp = str(forecast_period)
    ordered: list[str] = [fp]
    if "_" not in fp:
        return list(dict.fromkeys(ordered))
    base, step_s = fp.rsplit("_", 1)
    if step_s.isdigit():
        try:
            tp0 = TimePeriod.from_id(base)
            if tp0 is None:
                raise ValueError(f"Unsupported period {base}")
            td = tp0.time_delta
            if td is None:
                raise ValueError(f"Unsupported time delta for period {base}")
            shifted = (tp0 + td * int(step_s)).id
            ordered.append(str(shifted))
        except Exception:
            logger.debug("Could not derive shifted period from %s / %s", fp, base, exc_info=True)
        ordered.append(base)
    else:
        ordered.append(base)
    return list(dict.fromkeys(ordered))


def period_df_for_forecast(loc_df: pd.DataFrame, period_col: str, forecast_period: str) -> pd.DataFrame:
    if loc_df.empty or period_col not in loc_df.columns:
        return pd.DataFrame()
    s_series = loc_df[period_col]
    for cand in covariate_period_candidates(forecast_period):
        cstr = str(cand).strip()
        str_match = s_series.astype(str).str.strip() == cstr
        if str_match.any():
            return loc_df.loc[str_match].iloc[:1]
        ym_cand = year_month_from_any(cand)
        if ym_cand is None:
            continue
        idx = [i for i in range(len(loc_df)) if year_month_from_any(s_series.iloc[i]) == ym_cand]
        if idx:
            return loc_df.iloc[[idx[0]]]
    return pd.DataFrame()


def _parse_row_period(raw: Any) -> Month | Week | None:
    s = str(raw).strip()
    if not s:
        return None
    key = s
    if "_" in s:
        base, suf = s.rsplit("_", 1)
        if suf.isdigit():
            key = base
    try:
        if any(x in key for x in ("-W", "-S", "SunW")) or ("W" in key and not key.isdigit()):
            parsed = TimePeriod.from_id(clean_timestring(key))
        else:
            parsed = TimePeriod.from_id(key)
        if isinstance(parsed, Month | Week):
            return parsed
        return None
    except Exception:
        return None


def target_signature(forecast_period: str) -> tuple[str, int, int] | None:
    """Calendar period for covariates at this forecast id.

    For ``YYYYMM_k`` (monthly origin + horizon step k), the target calendar month is
    ``origin + k`` months. Step 1 is one month ahead of the base, step 2 is two months
    ahead, etc. (e.g. ``202405_3`` → August 2024 = May + 3).
    """
    fp = str(forecast_period)
    if "_" in fp:
        base, step_s = fp.rsplit("_", 1)
        if step_s.isdigit():
            try:
                tp0 = TimePeriod.from_id(base)
                if isinstance(tp0, Month):
                    td = tp0.time_delta
                    target_tp = tp0 + td * int(step_s)
                    ym = year_month_from_any(str(target_tp.id))
                    if ym is not None:
                        return ("month", ym[0], ym[1])
                if isinstance(tp0, Week):
                    td = tp0.time_delta
                    target_tp = tp0 + td * int(step_s)
                    if isinstance(target_tp, Week):
                        return ("week", target_tp.year, int(target_tp.week))
            except Exception:
                logger.debug("Horizon shift failed for forecast period %s", fp, exc_info=True)

    for cand in covariate_period_candidates(forecast_period):
        key = cand.rsplit("_", 1)[0] if "_" in cand and cand.rsplit("_", 1)[-1].isdigit() else cand
        tp = _parse_row_period(key)
        if isinstance(tp, Week):
            return ("week", tp.year, int(tp.week))
        ym = year_month_from_any(key)
        if ym is not None:
            return ("month", ym[0], ym[1])
    return None


def _historical_month_slice(
    loc_df: pd.DataFrame, period_col: str, target_year: int, target_month: int
) -> tuple[pd.DataFrame, list[int]]:
    prior_idx: list[int] = []
    any_year_idx: list[int] = []
    years_prior: list[int] = []
    years_any: list[int] = []
    for i in range(len(loc_df)):
        ym = year_month_from_any(loc_df[period_col].iloc[i])
        if ym is None or ym[1] != target_month:
            continue
        yrow = ym[0]
        any_year_idx.append(i)
        years_any.append(yrow)
        if yrow < target_year:
            prior_idx.append(i)
            years_prior.append(yrow)
    if prior_idx:
        return loc_df.iloc[prior_idx], sorted(set(years_prior))
    if any_year_idx:
        return loc_df.iloc[any_year_idx], sorted(set(years_any))
    return pd.DataFrame(), []


def _historical_week_slice(
    loc_df: pd.DataFrame, period_col: str, target_year: int, target_week: int
) -> tuple[pd.DataFrame, list[int]]:
    prior_idx: list[int] = []
    any_year_idx: list[int] = []
    years_prior: list[int] = []
    years_any: list[int] = []
    for i in range(len(loc_df)):
        tp = _parse_row_period(loc_df[period_col].iloc[i])
        if not isinstance(tp, Week) or int(tp.week) != int(target_week):
            continue
        any_year_idx.append(i)
        year = int(tp.year)
        years_any.append(year)
        if year < target_year:
            prior_idx.append(i)
            years_prior.append(year)
    if prior_idx:
        return loc_df.iloc[prior_idx], sorted(set(years_prior))
    if any_year_idx:
        return loc_df.iloc[any_year_idx], sorted(set(years_any))
    return pd.DataFrame(), []


def _aggregate_features(sub: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    row: dict[str, float] = {}
    for f in feature_names:
        if f in sub.columns and not sub.empty:
            col = pd.to_numeric(sub[f], errors="coerce")
            if col.notna().any():
                row[f] = float(col.mean())
            else:
                row[f] = float("nan")
        else:
            row[f] = float("nan")
    return row


def _provenance_dataset_match(matched_period: str) -> dict[str, Any]:
    return {
        "source": "dataset_match",
        "matchedPeriod": matched_period,
        "detail": "Covariates taken from the dataset row for the matched forecast period.",
    }


def resolve_covariate_row(
    loc_df: pd.DataFrame,
    period_col: str,
    feature_names: list[str],
    forecast_period: str,
    org_unit: str,
    global_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, Any]]:
    if loc_df.empty:
        fb = global_df.tail(1)
        row = _aggregate_features(fb, feature_names)
        return row, {
            "source": "last_available_row",
            "detail": "No rows for this location in the dataset; using the last available row in the full dataset.",
        }

    if not period_col or period_col not in loc_df.columns:
        fb = loc_df.tail(1) if not loc_df.empty else global_df.tail(1)
        row = _aggregate_features(fb, feature_names)
        return row, {
            "source": "last_available_row",
            "detail": "No usable period column for this location; covariates use the last available row.",
        }

    period_df = period_df_for_forecast(loc_df, period_col, forecast_period)
    if not period_df.empty:
        cand = str(period_df[period_col].iloc[0])
        row = _aggregate_features(period_df, feature_names)
        return row, _provenance_dataset_match(cand)

    sig = target_signature(forecast_period)
    if sig is not None and period_col in loc_df.columns and not loc_df.empty:
        kind, y, m_or_w = sig
        if kind == "month":
            sub, years_used = _historical_month_slice(loc_df, period_col, y, m_or_w)
            if not sub.empty:
                row = _aggregate_features(sub, feature_names)
                mon_name = calendar.month_name[m_or_w]
                prov = {
                    "source": "historical_same_month_mean",
                    "aggregate": "mean",
                    "targetYear": y,
                    "calendarMonth": m_or_w,
                    "nRowsAveraged": len(sub),
                    "yearsUsed": years_used,
                    "detail": (
                        f"Covariates are the mean for {mon_name} at this location "
                        f"across {len(sub)} dataset row(s) from year(s) {years_used}. "
                        f"No exact row for the forecast period; prior-year same-month values were used."
                    ),
                }
                logger.warning(
                    "No dataset row for org_unit=%s period=%s; using historical same-month mean (%d rows, years=%s).",
                    org_unit,
                    forecast_period,
                    len(sub),
                    years_used,
                )
                return row, prov
        else:
            sub, years_used = _historical_week_slice(loc_df, period_col, y, m_or_w)
            if not sub.empty:
                row = _aggregate_features(sub, feature_names)
                prov = {
                    "source": "historical_same_week_mean",
                    "aggregate": "mean",
                    "targetYear": y,
                    "isoWeek": m_or_w,
                    "nRowsAveraged": len(sub),
                    "yearsUsed": years_used,
                    "detail": (
                        f"Covariates are the mean for ISO week {m_or_w} at this location "
                        f"across {len(sub)} dataset row(s) from year(s) {years_used}. "
                        f"No exact row for the forecast period; prior-year same-week values were used."
                    ),
                }
                logger.warning(
                    "No dataset row for org_unit=%s period=%s; using historical same-week mean (%d rows).",
                    org_unit,
                    forecast_period,
                    len(sub),
                )
                return row, prov

    fb = loc_df.tail(1) if not loc_df.empty else global_df.tail(1)
    row = _aggregate_features(fb, feature_names)
    prov = {
        "source": "last_available_row",
        "detail": (
            "No matching or historical period in the dataset for this location; "
            "covariates use the last available row (may repeat across forecast steps)."
        ),
    }
    logger.warning(
        "No dataset row for org_unit=%s period=%s (tried %s); using last available row as fallback.",
        org_unit,
        forecast_period,
        covariate_period_candidates(forecast_period),
    )
    return row, prov
