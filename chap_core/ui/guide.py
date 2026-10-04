"""The decisions behind the guided "find the best model" flow, kept free of Streamlit so they can be tested:
what a dataset holds, which models fit it, which test settings it allows, and which model won."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from chap_core.services.dataset_validation import ValidationIssue
    from chap_core.services.model_marketplace import MarketplaceModel

# Columns every Chap dataset has; everything else is a covariate a model may use.
STRUCTURAL_COLUMNS = {"time_period", "location", "disease_cases", "parent"}

# The plain-language choices of step 3, mapped to `chap eval` backtest settings.
HORIZONS = {1: "1 month", 3: "3 months", 6: "6 months"}
THOROUGHNESS = {"Quick": 3, "Normal": 7, "Thorough": 12}


@dataclass(frozen=True)
class DatasetSummary:
    locations: int
    period_type: str | None  # "monthly", "weekly", or None when the periods are not recognised
    periods: list[str]
    covariates: list[str]
    has_polygons: bool

    @property
    def unit(self) -> str:
        return "week" if self.period_type == "weekly" else "month"


def summarize_dataset(csv_path: Path) -> DatasetSummary:
    """What a dataset holds, as the guide describes it."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    periods = sorted(df["time_period"].astype(str).unique()) if "time_period" in df.columns else []
    return DatasetSummary(
        locations=df["location"].nunique() if "location" in df.columns else 0,
        period_type=period_type(periods),
        periods=periods,
        covariates=[column for column in df.select_dtypes("number").columns if column not in STRUCTURAL_COLUMNS],
        has_polygons=csv_path.with_suffix(".geojson").exists(),
    )


def period_type(periods: list[str]) -> str | None:
    """Monthly (2010-01) or weekly (2010W01, 2010-W01) periods, judged from the first one."""
    if not periods:
        return None
    if re.fullmatch(r"\d{4}-?W\d{1,2}", periods[0]):
        return "weekly"
    if re.fullmatch(r"\d{4}-\d{2}", periods[0]):
        return "monthly"
    return None


def model_fit(model: MarketplaceModel, data: DatasetSummary) -> str | None:
    """Why a marketplace model cannot use this dataset, or None when it fits."""
    if model.kind != "model":
        return "is a template for building models, not a model to forecast with"
    period_types = model.compatibility.period_types
    if data.period_type and period_types and data.period_type not in period_types:
        return f"needs {' or '.join(period_types)} data, and this data is {data.period_type}"
    missing = [name for name in model.covariates.required if name not in data.covariates]
    if missing:
        columns = ", ".join(f"`{name}`" for name in missing)
        return f"needs {'a ' + columns + ' column' if len(missing) == 1 else 'the columns ' + columns}, which this data does not have"
    if model.compatibility.requires_geo and not data.has_polygons:
        return "needs a map of the regions, and this data has none"
    return None


def fit_reason(model: MarketplaceModel, data: DatasetSummary) -> str:
    """Why a model that fits does so, in one line."""
    used = [name for name in [*model.covariates.required, *model.covariates.defaults] if name in data.covariates]
    period = f"{data.period_type} data" if data.period_type else "this data"
    if not used:
        return f"Fits: {period}, needs nothing but the case counts"
    names = [name.replace("_", " ") for name in used]
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return f"Fits: {period}, uses {listed}, all in your data"


def horizon_limits(models: list[MarketplaceModel]) -> tuple[int, int]:
    """The forecast horizons every chosen model supports."""
    low = max([m.compatibility.min_prediction_periods or 1 for m in models] or [1])
    high = min([m.compatibility.max_prediction_periods or 100 for m in models] or [100])
    return max(low, 1), high


def issue_line(issue: ValidationIssue) -> str:
    """A validation issue in one line, with where it is and the first periods it concerns."""
    periods = issue.time_periods or []
    listed = ", ".join(periods[:5]) + (f" and {len(periods) - 5} more" if len(periods) > 5 else "")
    where = ", ".join(part for part in (issue.location, listed) if part)
    return issue.message + (f" ({where})" if where else "")


def kept_choice(saved, options, preferred):
    """The saved answer if it is still offered, else the preferred one, else the first option."""
    for choice in (saved, preferred):
        if choice in options:
            return choice
    return next(iter(options), None)


def best_model(metrics: pd.DataFrame) -> pd.Series:
    """The evaluation with the lowest CRPS: the score that also rewards a model for its uncertainty."""
    return metrics.sort_values("crps").iloc[0]
