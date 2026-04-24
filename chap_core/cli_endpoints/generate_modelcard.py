import json
import logging
from pathlib import Path
from typing import Annotated, Any, cast

import altair as alt
import xarray as xr
from cyclopts import Parameter

from chap_core.assessment.backtest_plots import create_plot_from_evaluation
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import (
    Coverage25_75Metric,
    CRPSNormMetric,
    MAPEMetric,
    RMSEMetric,
    compute_all_aggregated_metrics_from_backtest,
)
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.plotting.evaluation_plot import (
    MetricByTimePeriodV2Mean,
    MetricMapV2,
    RegionalMetricDistributionPlot,
    make_plot_from_evaluation_object,
)

try:
    from chap_core import __version__ as CHAP_VERSION
except ImportError:
    CHAP_VERSION = "unknown"

MISSING = "More Information Needed"

logger = logging.getLogger(__name__)

PLACEHOLDER_METADATA_VALUES = {
    "display_name": ModelTemplateMetaData.model_fields["display_name"].default,
    "description": ModelTemplateMetaData.model_fields["description"].default,
    "author_note": ModelTemplateMetaData.model_fields["author_note"].default,
    "author": ModelTemplateMetaData.model_fields["author"].default,
}


def is_url(path_or_url: str | Path) -> bool:
    """Detect if input is a URL or a directory path."""
    return str(path_or_url).startswith(("http://", "https://"))


def format_author_assessed_status(status: str | None) -> str:
    """Render status as an HTML color badge with text fallback."""
    if not status:
        return MISSING

    normalized_status = status.lower()
    status_to_color = {
        "gray": "#6b7280",
        "red": "#d7263d",
        "orange": "#e86208",
        "yellow": "#fff200",
        "green": "#2e8b57",
    }

    status_to_explanation = {
        "gray": "Not intended for use, or deprecated/meant for legacy use only.",
        "red": "Highly experimental prototype - not at all validated and only meant for early experimentation",
        "orange": "Has seen promise on limited data, needs manual configuration and careful evaluation",
        "yellow": "Ready for more rigorous testing",
        "green": "Validated, ready for use",
    }

    color = status_to_color.get(normalized_status)
    explanation = status_to_explanation.get(normalized_status)
    if not color:
        return status
    return f'<span style="color: {color}; font-weight: 600;">&#9679; {normalized_status} </span>({explanation})'


def _safe_parse_model_info(model_info_json: str) -> ModelTemplateConfigV2 | None:
    if not model_info_json:
        return None
    try:
        return ModelTemplateConfigV2.model_validate_json(model_info_json)
    except (ValueError, TypeError):
        return None


def _parse_model_config(model_config_raw: object) -> dict[str, Any]:
    if isinstance(model_config_raw, str) and model_config_raw:
        try:
            parsed = json.loads(model_config_raw)
        except json.JSONDecodeError:
            logger.error("An error occured when parsing model config")
            return {}
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
        logger.warning("The NetCDF model config is an invalid structure")
        return {}
    if isinstance(model_config_raw, dict):
        return cast("dict[str, Any]", model_config_raw)
    logger.warning("The NetCDF file has no model_configuration attr")
    return {}


def _load_dataset_attrs(evaluation_path: Path) -> dict[str, Any]:
    with xr.open_dataset(evaluation_path) as ds:
        raw_model_info = ds.attrs.get("model_info", "")
        model_info_json = raw_model_info if isinstance(raw_model_info, str) else ""

        return {
            "model_name": ds.attrs.get("model_name"),
            "model_info": _safe_parse_model_info(model_info_json),
            "model_version_attr": ds.attrs.get("model_version"),
            "created_date_attr": ds.attrs.get("created_date"),
            "model_config": _parse_model_config(ds.attrs.get("model_configuration")),
            "historical_context_periods": int(ds.attrs.get("historical_context_periods", 0)),
        }


def _normalize_metadata_value(value: str | None, field_name: str) -> str | None:
    if not value:
        return None

    stripped_value = value.strip()
    if not stripped_value:
        return None

    if stripped_value == PLACEHOLDER_METADATA_VALUES.get(field_name):
        return None

    return stripped_value


def _save_evaluation_plots(evaluation: Evaluation, output_dir: Path, geojson_path: Path | None) -> None:
    evaluation_plot = create_plot_from_evaluation("evaluation_plot", evaluation)
    evaluation_plot.save(output_dir / "eval_plot.png", scale_factor=2.0)
    evaluation_plot.save(output_dir / "eval_plot.html", scale_factor=2.0)

    detailed_rmse_plot = make_plot_from_evaluation_object(
        evaluation, RegionalMetricDistributionPlot, RMSEMetric()
    ).properties(title="RMSE distribution by Region")
    detailed_rmse_plot.save(output_dir / "detailedRMSE_plot.png", scale_factor=2.0)
    detailed_rmse_plot.save(output_dir / "detailedRMSE_plot.html", scale_factor=2.0)

    detailed_mape_plot = make_plot_from_evaluation_object(
        evaluation, RegionalMetricDistributionPlot, MAPEMetric()
    ).properties(title="MAPE distribution by Region")
    detailed_mape_plot.save(output_dir / "detailedMAPE_plot.png", scale_factor=2.0)
    detailed_mape_plot.save(output_dir / "detailedMAPE_plot.html", scale_factor=2.0)

    is_within_25th_75th_detailed_plot = make_plot_from_evaluation_object(
        evaluation, MetricByTimePeriodV2Mean, Coverage25_75Metric()
    ).properties(title="Within 25-75 Percentile")
    is_within_25th_75th_detailed_plot.save(output_dir / "isWithin25th75hDetailed_plot.png", scale_factor=2.0)
    is_within_25th_75th_detailed_plot.save(output_dir / "isWithin25th75hDetailed_plot.html", scale_factor=2.0)

    detailed_crps_norm_plot = make_plot_from_evaluation_object(
        evaluation, MetricByTimePeriodV2Mean, CRPSNormMetric()
    ).properties(title="Detailed Normalized CRPS")
    detailed_crps_norm_plot.save(output_dir / "detailedCRPSNorm_plot.png", scale_factor=2.0)
    detailed_crps_norm_plot.save(output_dir / "detailedCRPSNorm_plot.html", scale_factor=2.0)

    if geojson_path:
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))

        outline = (
            alt.Chart(alt.Data(values=geojson["features"]))
            .mark_geoshape(fill=None, stroke="#374151", strokeWidth=0.5)
            .project(type="equirectangular")
        )

        rmse_map_plot = (
            make_plot_from_evaluation_object(evaluation, MetricMapV2, RMSEMetric(), geojson)
            .properties(
                title="Aggregate RMSE by region",
            )
            .encode(color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), legend=alt.Legend(title="RMSE")))
        )

        mape_map_plot = (
            make_plot_from_evaluation_object(evaluation, MetricMapV2, MAPEMetric(), geojson)
            .properties(
                title="Aggregate MAPE by region",
            )
            .encode(color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), legend=alt.Legend(title="MAPE")))
        )

        rmse_map_plot = rmse_map_plot + outline
        rmse_map_plot.save(output_dir / "rmse_map.png", scale_factor=2.0)
        rmse_map_plot.save(output_dir / "rmse_map.html", scale_factor=2.0)

        mape_map_plot = mape_map_plot + outline
        mape_map_plot.save(output_dir / "mape_map.png", scale_factor=2.0)
        mape_map_plot.save(output_dir / "mape_map.html", scale_factor=2.0)


def _build_results_summary(backtest) -> str:
    metrics = compute_all_aggregated_metrics_from_backtest(backtest)
    return "\n".join(
        [
            f"Ratio above truth: {metrics.get('ratio_above_truth')}\n",
            f"CRPS: {metrics.get('crps')}\n",
            f"CRPS Normalized: {metrics.get('crps_norm')}\n",
            f"Example metric: {metrics.get('example_metric')}\n",
            f"RMSE (aggregate): {metrics.get('rmse')}\n",
            f"MAE (aggregate): {metrics.get('mae')}\n",
            f"Coverage within 10-90%: {metrics.get('coverage_10_90')}\n",
            f"Coverage within 25-75%: {metrics.get('coverage_25_75')}\n",
            f"Sample count: {metrics.get('sample_count')}\n",
        ]
    )


def generate_modelcard(
    evaluation_path: Annotated[
        Path,
        Parameter(help="Path to NetCDF file containing evaluation data"),
    ],
    output_file: Annotated[
        Path,
        Parameter(help="Path or name of output Markdown file containing modelcard template"),
    ],
    geojson_path: Annotated[
        Path | None,
        Parameter(help="Path to GeoJSON file matching the regions of the evaluation dataset"),
    ] = None,
):
    """Generates a modelcard document and exports a resulting Markdown file and plots in PNG and interactive HTML.

    Alongside the standard metric summary and aggregate measures, the modelcard also shows a regional breakdown of
    RMSE and MAPE distribution.
    The completeness of the modelcard template is dependent on the amount of information in the models MLproject file.

    Optionally generates MAP based plots showing aggregate RMSE and MAPE given a geojson file.
    """

    attrs = _load_dataset_attrs(evaluation_path)

    model_name = attrs["model_name"]
    model_info: ModelTemplateConfigV2 = attrs["model_info"]
    model_version_attr = attrs["model_version_attr"]
    created_date_attr = attrs["created_date_attr"]
    model_config = attrs["model_config"]
    historical_context_periods = attrs["historical_context_periods"]

    hyperparameters = model_config.get("user_option_values", {})
    additional_covariates = model_config.get("additional_continuous_covariates", [])

    evaluation = Evaluation.from_file(evaluation_path)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    chap_version = CHAP_VERSION
    org_units = evaluation.get_org_units()
    split_periods = evaluation.get_split_periods()
    backtest = evaluation.to_backtest()
    created_date = backtest.created or created_date_attr

    meta_data = model_info.meta_data if model_info else None
    author = _normalize_metadata_value(meta_data.author, "author") if meta_data else None
    organization = meta_data.organization if meta_data and meta_data.organization else None
    author_assessed_status = meta_data.author_assessed_status.value if meta_data else None
    organization_logo_url = meta_data.organization_logo_url if meta_data and meta_data.organization_logo_url else None
    citation_info = meta_data.citation_info if meta_data and meta_data.citation_info else None
    contact_email = meta_data.contact_email if meta_data and meta_data.contact_email else None
    display_name = _normalize_metadata_value(meta_data.display_name, "display_name") if meta_data else None
    author_note = _normalize_metadata_value(meta_data.author_note, "author_note") if meta_data else None
    description = _normalize_metadata_value(meta_data.description, "description") if meta_data else None

    _save_evaluation_plots(evaluation, output_dir, geojson_path)
    results_summary = _build_results_summary(backtest)

    output_path = output_file.with_suffix(".md")

    md: list[str] = []
    display_name = display_name or (
        model_info.name if model_info and model_info.name else (model_name or "Unknown Model")
    )
    md.append(f"# Model card for: {display_name}")
    md.append("")
    if author_note:
        md.append(author_note)
        md.append("")

    md.append("## Model details")
    md.append("")

    md.append("### Model description")
    md.append("")
    if description:
        md.append(description)

    # Not in HF template, but important regardless:
    model_version = model_info.version if model_info and model_info.version else (model_version_attr or MISSING)
    md.append(f"- **Model Version:** {model_version}")
    developed_by = ", ".join(value for value in [author, organization] if value) or MISSING
    md.append(f"- **Developed by:** {developed_by}".rstrip())
    md.append(f"- **Organization URL:** {MISSING}")
    md.append(f"- **Author assessed status:** {format_author_assessed_status(author_assessed_status)}")
    md.append(f"- **Funded by [optional]:** {MISSING}")
    md.append(f"- **Shared by [optional]:** {MISSING}")
    md.append(f"- **Model type:** {MISSING}")
    md.append(f"- **License:** {MISSING}")
    md.append(f"- **Finetuned from model: [optional]** {MISSING}")
    if organization_logo_url and is_url(organization_logo_url):
        md.append(
            f'<div><img src="{organization_logo_url}" alt="Organization logo" width="120" style="display: block;" align="right"/></div>'
        )
        md.append("")

    md.append("### Model Sources [optional]")
    source_url = (
        model_info.source_url
        if model_info and model_info.source_url
        else (model_name if model_name and is_url(model_name) else MISSING)
    )
    md.append(f"- **Repository:** {source_url}")
    md.append(f"- **Paper [optional]:** {MISSING}")
    md.append(f"- **Demo[optional]:** {MISSING}")

    if chap_version:
        md.append(f"- **CHAP version:** `{chap_version}`")

    md.append("")
    md.append("## Uses")
    md.append("")
    md.append("### Direct use: ")
    md.append("### Downstream use (optional): ")
    md.append("### Out-of-scope use: ")
    md.append("")

    md.append("## Bias, Risks and Limitations")
    md.append("")
    md.append("### Recommendations: ")
    md.append("")

    md.append("## Training details")
    md.append("")

    md.append("### Training data")
    md.append("")
    md.append(f"- Created date: {created_date}")
    md.append(f"- Number of organization units: {len(org_units)}")
    if model_info:
        md.append(f"- Supported period type: `{model_info.supported_period_type.value}`")
        md.append(f"- Required covariates: {', '.join(model_info.required_covariates) or 'none'}")
        md.append(f"- Allow free additional covariates: `{model_info.allow_free_additional_continuous_covariates}`")
    else:
        md.append(f"- Supported period type: {MISSING}")
        md.append(f"- Required covariates: {MISSING}")
        md.append(f"- Allow free additional covariates: {MISSING}")
    md.append("")
    md.append("### Training procedure")
    md.append("")
    md.append("#### Preprocessing [optional]")
    md.append("- Splitting historical data into multiple train/test sets using rolling-origin backtesting")
    md.append(f"- Split periods: {', '.join(map(str, split_periods))}")
    md.append(f"- Historical context periods: {historical_context_periods}")
    md.append("")
    md.append("#### Training Hyperparameters")
    if isinstance(hyperparameters, dict) and hyperparameters:
        md.append("- **User option values:**")
        md.extend([f"  - `{key}`: `{hyperparameters[key]}`" for key in sorted(hyperparameters)])
    else:
        md.append(f"- **User option values:** {MISSING}")

    if isinstance(additional_covariates, list) and additional_covariates:
        md.append(f"- **Additional covariates:** {', '.join(str(v) for v in additional_covariates)}")
    else:
        md.append("- **Additional covariates:** none")

    md.append("- **Training regime:**")
    md.append("#### Speeds, Sizes, Times [optional]")

    md.append("## Evaluation")
    md.append("")
    md.append("### Testing Data, Factors & Metrics")
    md.append("")
    md.append("#### Testing Data")
    md.append("")
    md.append("#### Factors")
    md.append("")
    md.append("#### Metrics")
    md.append("")
    md.append("##### Aggregate Metrics")
    md.append("")
    md.append("Metrics Summary")
    md.append("")
    md.append(results_summary)
    md.append("")
    md.append("#### Forecast vs Observations")
    md.append("")
    md.append("![Evaluation plot](eval_plot.png)")
    md.append("")
    md.append("[Evaluation interactive chart](eval_plot.html)")
    md.append("")
    md.append("#### Coverage within 25-75% by time period\n\n![Coverage 25-75](isWithin25th75hDetailed_plot.png)\n")
    md.append("[Coverage interactive chart](isWithin25th75hDetailed_plot.html)")
    md.append("#### CRPS Normalized by time period\n\n![CRPS normalized](detailedCRPSNorm_plot.png)\n")
    md.append("[CRPS interactive chart](detailedCRPSNorm_plot.html)")
    md.append("#### Regional breakdown")
    if geojson_path:
        md.append("#### Aggregate RMSE Map by Region \n\n![Aggregate RMSE Map by region](rmse_map.png)\n")
        md.append("")
        md.append("[Interactive RMSE Map chart](rmse_map.html)")
        md.append("")
    md.append("#### Regional RMSE distribution \n\n![RMSE by region](detailedRMSE_plot.png)\n")
    md.append("")
    md.append("[RMSE interactive chart](detailedRMSE_plot.html)")
    if geojson_path:
        md.append("")
        md.append("#### Aggregate MAPE Map by Region \n\n![Aggregate MAPE Map by region](mape_map.png)\n")
        md.append("")
        md.append("[Interactive MAPE Map chart](mape_map.html)")
    md.append("")
    md.append("#### Regional MAPE distribution\n\n![MAPE by region](detailedMAPE_plot.png)\n")
    md.append("")
    md.append("[MAPE interactive chart](detailedMAPE_plot.html)")
    md.append("### Results")
    md.append("")
    md.append("#### Summary")

    md.append("## Model examination [optional]")

    md.append("## Environmental Impact")
    md.append("")
    md.append(
        "Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700)."
    )
    md.append("- **Hardware Type:**")
    md.append("- **Hours used:**")
    md.append("- **Cloud Provider:**")
    md.append("- **Compute Region:**")
    md.append("- **Carbon Emitted:**")

    md.append("## Technical Specifications [optional]")
    md.append("")
    md.append("### Model Architecture and Objective")
    md.append("### Compute Infrastructure")
    md.append("#### Hardware")
    md.append("#### Software")

    md.append("## Citation [optional]")
    md.append("")
    md.append("**BibTeX:**")
    md.append("")
    md.append("**APA:**")
    md.append("")
    if citation_info:
        md.append(citation_info)

    md.append("## Glossary [optional]")
    md.append("")

    md.append("## More information [optional]")
    md.append("")

    md.append("## Model Card Authors [optional]")
    md.append("")

    md.append("## Model card contact")
    md.append("")
    if contact_email:
        md.append(f"- Contact email: {contact_email}")
    else:
        md.append("- Contact email: ")

    output_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def register_commands(app):
    app.command()(generate_modelcard)
