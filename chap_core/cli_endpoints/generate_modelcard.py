from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import TYPE_CHECKING, Annotated, Any, cast

from cyclopts import Parameter

if TYPE_CHECKING:
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.database.tables import BackTest
    from chap_core.external.model_configuration import ModelTemplateConfigV2

CHAP_VERSION = importlib.metadata.version("chap-core")

MISSING = "More Information Needed"

logger = logging.getLogger(__name__)


@functools.cache
def _placeholder_metadata_values() -> dict[str, str | None]:
    """Defaults from ModelTemplateMetaData fields, used to detect un-edited placeholders."""
    from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData

    return {
        "display_name": ModelTemplateMetaData.model_fields["display_name"].default,
        "description": ModelTemplateMetaData.model_fields["description"].default,
        "author_note": ModelTemplateMetaData.model_fields["author_note"].default,
        "author": ModelTemplateMetaData.model_fields["author"].default,
    }


@dataclass(frozen=True)
class ModelCardContext:
    display_name: str
    author_note: str | None
    description: str | None
    model_version: str
    developed_by: str
    organization_logo_url: str | None
    author_assessed_status: str | None
    source_url: str
    chap_version: str
    created_date: str | None
    org_units_count: int
    supported_period_type: str
    required_covariates: str
    allow_free_additional_continuous_covariates: str
    split_periods: str
    historical_context_periods: int
    user_option_value_lines: list[str]
    additional_covariates: str
    results_summary: str
    contact_email: str | None
    citation_info: str | None
    include_geojson_maps: bool
    documentation_url: str


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
    from chap_core.external.model_configuration import ModelTemplateConfigV2

    try:
        return ModelTemplateConfigV2.model_validate_json(model_info_json)
    except (ValueError, TypeError):
        return None


def _parse_model_config(model_config_raw: object) -> dict[str, Any]:
    if isinstance(model_config_raw, str) and model_config_raw:
        try:
            parsed = json.loads(model_config_raw)
        except json.JSONDecodeError:
            logger.error("An error occurred when parsing model config")
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
    import xarray as xr

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

    if stripped_value == _placeholder_metadata_values().get(field_name):
        return None

    return stripped_value


def _save_evaluation_plots(evaluation: Evaluation, output_dir: Path, geojson_path: Path | None) -> None:
    import altair as alt

    from chap_core.assessment.backtest_plots import create_plot_from_evaluation
    from chap_core.assessment.metric_plots.metric_map import MetricMapV2
    from chap_core.assessment.metric_plots.regional_distribution import RegionalMetricDistributionPlot
    from chap_core.assessment.metric_plots.time_period_mean import MetricByTimePeriodV2Mean
    from chap_core.assessment.metrics import (
        Coverage25_75Metric,
        CRPSNormMetric,
        MAPEMetric,
        RMSEMetric,
    )
    from chap_core.plotting.evaluation_plot import make_plot_from_evaluation_object

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
        logger.info("Creating map plots using geojson file")
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        if not isinstance(geojson, dict) or "features" not in geojson:
            raise ValueError(f"Invalid GeoJSON at {geojson_path}: expected a 'features' key.")

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


def _build_results_summary(backtest: BackTest) -> str:

    from chap_core.assessment.metrics import compute_all_aggregated_metrics_from_backtest

    metrics = compute_all_aggregated_metrics_from_backtest(backtest)
    return "\n".join(
        [
            f"Ratio above truth: {metrics.get('ratio_above_truth')}\n",
            f"CRPS: {metrics.get('crps')}\n",
            f"CRPS Normalized: {metrics.get('crps_norm')}\n",
            f"RMSE (aggregate): {metrics.get('rmse')}\n",
            f"MAE (aggregate): {metrics.get('mae')}\n",
            f"Coverage within 10-90%: {metrics.get('coverage_10_90')}\n",
            f"Coverage within 25-75%: {metrics.get('coverage_25_75')}\n",
            f"Sample count: {metrics.get('sample_count')}\n",
        ]
    )


def _render_title_section(context: ModelCardContext) -> list[str]:
    lines = [f"# Model card for: {context.display_name}"]
    lines.append("")
    if context.author_note:
        lines.append(context.author_note)
        lines.append("")
    return lines


def _render_model_details_section(context: ModelCardContext) -> list[str]:
    lines = ["## Model details", "", "### Model description", ""]
    if context.description:
        lines.append(context.description)

    lines.extend(
        [
            f"- **Model Version:** {context.model_version}",
            f"- **Developed by:** {context.developed_by}",
            f"- **Author assessed status:** {format_author_assessed_status(context.author_assessed_status)}",
            f"- **Funded by [optional]:** {MISSING}",
            f"- **Shared by [optional]:** {MISSING}",
            f"- **Model type:** {MISSING}",
            f"- **License:** {MISSING}",
            f"- **Finetuned from model: [optional]** {MISSING}",
        ]
    )

    if context.organization_logo_url and is_url(context.organization_logo_url):
        lines.extend(
            [
                f'<div><img src="{context.organization_logo_url}" alt="Organization logo" width="120" style="display: block;" align="right"/></div>',
                "",
            ]
        )

    lines.extend(
        [
            "### Model Sources [optional]",
            f"- **Repository:** {context.source_url}",
            f"- **Paper [optional]:** {MISSING}",
            f"- **Demo[optional]:** {MISSING}",
        ]
    )

    if context.chap_version:
        lines.append(f"- **CHAP version:** `{context.chap_version}`")

    lines.extend(
        [
            "",
            "## Uses",
            "",
            "### Direct use: ",
            "### Downstream use (optional): ",
            "### Out-of-scope use: ",
            "",
            "## Bias, Risks and Limitations",
            "",
            "### Recommendations: ",
            "",
        ]
    )
    return lines


def _render_training_section(context: ModelCardContext) -> list[str]:
    lines = ["## Training details", "", "### Training data", ""]
    lines.append(f"- Created date: {context.created_date}")
    lines.append(f"- Number of organization units: {context.org_units_count}")
    lines.append(f"- Supported period type: {context.supported_period_type}")
    lines.append(f"- Required covariates: {context.required_covariates}")
    lines.append(f"- Allow free additional covariates: {context.allow_free_additional_continuous_covariates}")
    lines.extend(
        [
            "",
            "### Training procedure",
            "",
            "#### Preprocessing [optional]",
            "- Splitting historical data into multiple train/test sets using rolling-origin backtesting",
            f"- Split periods: {context.split_periods}",
            f"- Historical context periods: {context.historical_context_periods}",
            "",
            "#### Training Hyperparameters",
        ]
    )
    lines.extend(context.user_option_value_lines)
    lines.append(f"- **Additional covariates:** {context.additional_covariates}")
    lines.extend(["- **Training regime:**", "#### Speeds, Sizes, Times [optional]"])
    return lines


def _render_evaluation_section(context: ModelCardContext) -> list[str]:
    lines = [
        "## Evaluation",
        "",
        "### Testing Data, Factors & Metrics",
        "",
        "#### Testing Data",
        "",
        "#### Factors",
        "",
        "#### Metrics",
        "",
        "##### Aggregate Metrics",
        "",
        "Metrics Summary",
        "",
        context.results_summary,
        "",
        "#### Forecast vs Observations",
        "",
        "![Evaluation plot](eval_plot.png)",
        "",
        "[Evaluation interactive chart](eval_plot.html)",
        "",
        "#### Coverage within 25-75% by time period\n\n![Coverage 25-75](isWithin25th75hDetailed_plot.png)\n",
        "[Coverage interactive chart](isWithin25th75hDetailed_plot.html)",
        "#### CRPS Normalized by time period\n\n![CRPS normalized](detailedCRPSNorm_plot.png)\n",
        "[CRPS interactive chart](detailedCRPSNorm_plot.html)",
        "#### Regional breakdown",
    ]

    if context.include_geojson_maps:
        lines.extend(
            [
                "#### Aggregate RMSE Map by Region \n\n![Aggregate RMSE Map by region](rmse_map.png)\n",
                "",
                "[Interactive RMSE Map chart](rmse_map.html)",
                "",
            ]
        )

    lines.extend(
        [
            "#### Regional RMSE distribution \n\n![RMSE by region](detailedRMSE_plot.png)\n",
            "",
            "[RMSE interactive chart](detailedRMSE_plot.html)",
        ]
    )

    if context.include_geojson_maps:
        lines.extend(
            [
                "",
                "#### Aggregate MAPE Map by Region \n\n![Aggregate MAPE Map by region](mape_map.png)\n",
                "",
                "[Interactive MAPE Map chart](mape_map.html)",
            ]
        )

    lines.extend(
        [
            "",
            "#### Regional MAPE distribution\n\n![MAPE by region](detailedMAPE_plot.png)\n",
            "",
            "[MAPE interactive chart](detailedMAPE_plot.html)",
            "### Results",
            "",
            "#### Summary",
        ]
    )
    return lines


def _render_additional_sections(context: ModelCardContext) -> list[str]:
    lines = ["## Model examination [optional]", "", "## Environmental Impact", ""]
    lines.append(
        "Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700)."
    )
    lines.extend(
        [
            "- **Hardware Type:**",
            "- **Hours used:**",
            "- **Cloud Provider:**",
            "- **Compute Region:**",
            "- **Carbon Emitted:**",
            "",
            "## Technical Specifications [optional]",
            "",
            "### Model Architecture and Objective",
            "### Compute Infrastructure",
            "#### Hardware",
            "#### Software",
            "",
            "## Citation [optional]",
            "",
            "**BibTeX:**",
            "",
            "**APA:**",
            "",
        ]
    )

    if context.citation_info:
        lines.append(context.citation_info)

    lines.extend(
        [
            "",
            "## Glossary [optional]",
            "",
            "## More information",
            "",
            f"- Documentation URL: {context.documentation_url}",
            "",
            "## Model Card Authors [optional]",
            "",
            "## Model card contact",
            "",
        ]
    )
    if context.contact_email:
        lines.append(f"- Contact email: {context.contact_email}")
    else:
        lines.append("- Contact email: ")
    return lines


def render_modelcard(context: ModelCardContext) -> str:
    md: list[str] = []
    md.extend(_render_title_section(context))
    md.extend(_render_model_details_section(context))
    md.extend(_render_training_section(context))
    md.extend(_render_evaluation_section(context))
    md.extend(_render_additional_sections(context))
    return "\n".join(md).rstrip() + "\n"


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
    from chap_core.assessment.evaluation import Evaluation

    logger.info(f"Generating Model Card from {evaluation_path}")

    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation file not found at: {evaluation_path}")

    if geojson_path and not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at: {geojson_path}")

    attrs = _load_dataset_attrs(evaluation_path)

    model_name = attrs["model_name"]
    model_info: ModelTemplateConfigV2 | None = attrs["model_info"]
    model_version_attr = attrs["model_version_attr"]
    created_date_attr = attrs["created_date_attr"]
    model_config = attrs["model_config"]
    historical_context_periods = attrs["historical_context_periods"]

    hyperparameters = model_config.get("user_option_values", {})
    additional_covariates = model_config.get("additional_continuous_covariates", [])

    evaluation = Evaluation.from_file(evaluation_path)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    org_units = evaluation.get_org_units()
    split_periods = evaluation.get_split_periods()
    backtest = evaluation.to_backtest()
    created_date = backtest.created or created_date_attr

    meta_data = model_info.meta_data if model_info else None
    author = _normalize_metadata_value(meta_data.author, "author") if meta_data else None
    organization = meta_data.organization if meta_data else None
    author_assessed_status = meta_data.author_assessed_status.value if meta_data else None
    organization_logo_url = meta_data.organization_logo_url if meta_data else None
    citation_info = meta_data.citation_info if meta_data else None
    contact_email = meta_data.contact_email if meta_data else None
    documentation_url = (meta_data.documentation_url if meta_data else None) or MISSING
    display_name = _normalize_metadata_value(meta_data.display_name, "display_name") if meta_data else None
    author_note = _normalize_metadata_value(meta_data.author_note, "author_note") if meta_data else None
    description = _normalize_metadata_value(meta_data.description, "description") if meta_data else None

    logger.info(f"Saving evaluation plots to {output_dir.absolute()}")
    _save_evaluation_plots(evaluation, output_dir, geojson_path)
    results_summary = _build_results_summary(backtest)

    model_version = model_info.version if model_info and model_info.version else (model_version_attr or MISSING)
    developed_by = ", ".join(value for value in [author, organization] if value) or MISSING
    source_url = (
        model_info.source_url
        if model_info and model_info.source_url
        else (model_name if model_name and is_url(model_name) else MISSING)
    )

    display_name = display_name or (
        model_info.name if model_info and model_info.name else (model_name or "Unknown Model")
    )
    user_option_value_lines = (
        ["- **User option values:**", *[f"  - `{key}`: `{hyperparameters[key]}`" for key in sorted(hyperparameters)]]
        if isinstance(hyperparameters, dict) and hyperparameters
        else [f"- **User option values:** {MISSING}"]
    )
    modelcard_context = ModelCardContext(
        display_name=display_name,
        author_note=author_note,
        description=description,
        model_version=model_version,
        developed_by=developed_by,
        organization_logo_url=organization_logo_url,
        author_assessed_status=author_assessed_status,
        source_url=source_url,
        chap_version=CHAP_VERSION,
        created_date=str(created_date) if created_date is not None else None,
        org_units_count=len(org_units),
        supported_period_type=f"`{model_info.supported_period_type.value}`" if model_info else MISSING,
        required_covariates=", ".join(model_info.required_covariates) or "none" if model_info else MISSING,
        allow_free_additional_continuous_covariates=(
            f"`{model_info.allow_free_additional_continuous_covariates}`" if model_info else MISSING
        ),
        split_periods=", ".join(map(str, split_periods)),
        historical_context_periods=historical_context_periods,
        user_option_value_lines=user_option_value_lines,
        additional_covariates=(
            ", ".join(str(v) for v in additional_covariates)
            if isinstance(additional_covariates, list) and additional_covariates
            else "none"
        ),
        results_summary=results_summary,
        contact_email=contact_email,
        citation_info=citation_info,
        include_geojson_maps=bool(geojson_path),
        documentation_url=documentation_url,
    )

    output_file.write_text(render_modelcard(modelcard_context), encoding="utf-8")
    logger.info(f"Model card written to {output_file}")


def register_commands(app):
    app.command()(generate_modelcard)
