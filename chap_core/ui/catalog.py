"""Which page in the UI covers which chap command, grouped as the sidebar shows them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandPage:
    command: str
    title: str
    icon: str
    notes: str = ""


# Commands with a hand-made page in chap_core/ui/views/, mapped to that view.
CUSTOM_VIEWS = {
    "validate": "data.py",
    "plot-dataset": "data.py",
    "eval": "evaluate.py",
    "export-metrics": "results.py",
    "plot-backtest": "results.py",
    "model schema": "models.py",
}

# Commands rendered by the generic form page, per sidebar section.
COMMAND_PAGES: dict[str, list[CommandPage]] = {
    "Models": [
        CommandPage("sanity-check-model", "Sanity check", ":material/health_and_safety:"),
    ],
    "Analysis": [
        CommandPage("causal", "Counterfactual analysis", ":material/compare_arrows:"),
        CommandPage(
            "causal build-counterfactual",
            "Build counterfactual",
            ":material/tune:",
            "Transformations are written as `column=expression`, one per line, where `x` is the current value: "
            "`rainfall=x*0.8`, `disease_cases=round(x*1.1)`, or `rainfall=seasonal_min` / `window_avg_max`.",
        ),
        CommandPage("explain-lime", "Explain predictions (LIME)", ":material/lightbulb:"),
        CommandPage("evaluate-ensemble", "Ensemble evaluation", ":material/hub:"),
        CommandPage(
            "preference-learn",
            "Preference learning",
            ":material/thumbs_up_down:",
            "The UI runs preference learning in `metric` decision mode; `visual` mode needs a terminal.",
        ),
        CommandPage("forecast", "Forecast (legacy)", ":material/trending_up:"),
        CommandPage("multi-forecast", "Multi-forecast (legacy)", ":material/stacked_line_chart:"),
    ],
    "Reports": [
        CommandPage("generate-modelcard", "Model card", ":material/badge:"),
        CommandPage("report", "Model report", ":material/description:"),
        CommandPage("generate-pdf-report", "PDF report", ":material/picture_as_pdf:"),
        CommandPage("plot-backtest", "Save backtest plot", ":material/save:"),
        CommandPage("export-metrics", "Export metrics", ":material/table_view:"),
        CommandPage("aggregate-eval", "Aggregate evaluation", ":material/account_tree:"),
    ],
    "Tools": [
        CommandPage("convert-request", "Convert request", ":material/swap_horiz:"),
        CommandPage("write-open-api-spec", "OpenAPI spec", ":material/api:"),
        CommandPage("test", "Self-test", ":material/build:"),
    ],
}


def covered_commands() -> set[str]:
    """Every command that has a page in the UI."""
    return set(CUSTOM_VIEWS) | {page.command for pages in COMMAND_PAGES.values() for page in pages}
