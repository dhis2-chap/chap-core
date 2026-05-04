import logging
from typing import Any

from chap_core.xai.covariate_fallback import target_signature, year_month_from_any

logger = logging.getLogger(__name__)


def norm_period_id(p: str) -> str:
    return str(p).strip().replace("-", "_")


def find_forecast_row_index(forecasts: list[Any], org_unit: str, period: str) -> int | None:
    """Index in ``forecasts`` for (org_unit, period).

    Normalizes ``-`` vs ``_``. Avoids matching ``startswith(calendar_month)``, which
    incorrectly mapped every horizon (e.g. ``202405_2``) to the first ``202405_*`` row.

    When ``period`` is a horizon-step id like ``202406_k``, and no forecast stores that
    literal period, we derive the target calendar month/week from ``target_signature``
    and look for a forecast whose stored period falls in that calendar period.
    """
    nreq = norm_period_id(period)
    for i, f in enumerate(forecasts):
        if f.org_unit == org_unit and norm_period_id(f.period) == nreq:
            return i

    unit_indices = [i for i, f in enumerate(forecasts) if f.org_unit == org_unit]
    if not unit_indices:
        return None

    if "_" in period:
        tail = period.rsplit("_", 1)[-1]
        if tail.isdigit():
            nbase = norm_period_id(period.rsplit("_", 1)[0])
            matches = [
                i
                for i in unit_indices
                if norm_period_id(forecasts[i].period).endswith("_" + tail)
                and norm_period_id(forecasts[i].period).rsplit("_", 1)[0] == nbase
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                logger.warning(
                    "Ambiguous period %s for org_unit=%s among %s; using index %s",
                    period,
                    org_unit,
                    [forecasts[i].period for i in matches],
                    matches[0],
                )
                return matches[0]

            # Horizon-step notation (e.g. "202406_2") with forecasts stored as plain
            # calendar periods (e.g. "202407"). Derive the target calendar month and
            # match against forecast periods.
            sig = target_signature(period)
            if sig is not None and sig[0] == "month":
                _, target_year, target_month = sig
                cal_matches = [
                    i for i in unit_indices if year_month_from_any(forecasts[i].period) == (target_year, target_month)
                ]
                if len(cal_matches) == 1:
                    return cal_matches[0]
                if len(cal_matches) > 1:
                    return cal_matches[0]

    if len(unit_indices) == 1:
        return unit_indices[0]

    logger.warning(
        "Period %s not found for org_unit=%s. Known periods: %s",
        period,
        org_unit,
        [forecasts[i].period for i in unit_indices[:12]],
    )
    return None
