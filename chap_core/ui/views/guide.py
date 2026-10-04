"""Find the best model for my data: a guided comparison in five steps.

Data, the models that fit it, two plain questions, a run that starts what it needs, and an answer in words.
The guide's progress lives in st.session_state["guide"], so leaving the page and coming back resumes it.
"""

from pathlib import Path
from typing import Any, Literal

import streamlit as st

from chap_core.ui.guide import (
    HORIZONS,
    THOROUGHNESS,
    answer_text,
    best_model,
    fit_reason,
    horizon_limits,
    issue_line,
    kept_choice,
    model_fit,
    summarize_dataset,
)
from chap_core.ui.guide_run import eval_args, stop_started
from chap_core.ui.models import ASSESSMENT, chaps_binary, list_services, marketplace_image
from chap_core.ui.services import (
    PUBLISHED_DATASETS,
    backtest_windows,
    fetch_published_dataset,
    format_cli_command,
    get_uploads_dir,
    save_upload,
)
from chap_core.ui.widgets import chaps_location, load_chaps_models, load_marketplace

STEPS = ["Your data", "Models", "Questions", "Run", "Answer"]
BADGE_COLORS: dict[str, Literal["green", "orange", "red", "gray"]] = {
    "good": "green",
    "warn": "orange",
    "bad": "red",
    "neutral": "gray",
}

guide: dict = st.session_state.setdefault("guide", {"step": 1})
pages = st.session_state["pages"]
marketplace = {m.id: m for m in load_marketplace()[0]}


def go(step: int) -> None:
    guide["step"] = step
    st.rerun()


def steps_bar() -> None:
    """Where the user is, with what they chose in the steps behind them."""
    done = {
        1: guide.get("dataset_name"),
        2: ", ".join(marketplace[i].display_name or i for i in guide.get("models", []) if i in marketplace),
        3: f"{guide.get('horizon')} ahead, {guide.get('splits')} tests" if guide.get("splits") else None,
    }
    cols = st.columns(len(STEPS))
    for number, (col, name) in enumerate(zip(cols, STEPS, strict=True), start=1):
        text = f"**{number} · {name}**" if number == guide["step"] else f"{number} · {name}"
        if number < guide["step"] and done.get(number):
            text += f"  \n:gray[{done[number]}]"
        col.markdown(text if number <= guide["step"] else f":gray[{text}]")
    st.progress(guide["step"] / len(STEPS))


st.caption("Find the best model for my data")
steps_bar()


# -- Step 1: data --


def choose_dataset() -> Path | None:
    uploads_dir = get_uploads_dir()
    earlier = sorted(uploads_dir.glob("*.csv")) if uploads_dir.exists() else []
    sources = ["Use an example", "Upload my own file"] + (["A file I used before"] if earlier else [])
    source = st.segmented_control(
        "Where is your data?",
        sources,
        default=sources[0],
        key="guide-source",
        help="A published example to try things out, your own CSV file, or a file you used here before.",
    )
    if source == "Use an example":
        name = st.selectbox(
            "Example",
            list(PUBLISHED_DATASETS),
            key="guide-example",
            help="Province-level monthly data from dhis2/climate-health-data, with a map of the regions.",
        )
        st.caption("Published in dhis2/climate-health-data, with a map of the regions.")
        with st.spinner("Downloading the dataset..."):
            path = fetch_published_dataset(uploads_dir, PUBLISHED_DATASETS[name])
        guide["dataset_name"] = name
        return path
    if source == "Upload my own file":
        cols = st.columns(2)
        csv_file = cols[0].file_uploader(
            "Your data, as a CSV file",
            type="csv",
            key="guide-upload",
            help="One row per region and month (or week): time_period, location, disease_cases, and any covariates.",
        )
        geojson = cols[1].file_uploader(
            "A map of the regions (optional GeoJSON)",
            type=["geojson", "json"],
            help="Lets Chap draw maps, and lets models that use region shapes run.",
        )
        if csv_file is None:
            return None
        path = save_upload(uploads_dir, csv_file.name, csv_file.getvalue())
        if geojson is not None:
            save_upload(uploads_dir, path.with_suffix(".geojson").name, geojson.getvalue())
        guide["dataset_name"] = path.name
        return path
    choice = st.selectbox(
        "Earlier file",
        earlier,
        format_func=lambda p: p.name,
        key="guide-earlier",
        help="Files you uploaded or downloaded here before, from the uploads folder.",
    )
    guide["dataset_name"] = choice.name
    return choice


def step_data() -> None:
    st.header("Which data do you want to forecast?")
    st.markdown(
        ":gray[A CSV file with one row per region and month (or week): the number of cases, and anything else "
        "you have, such as rainfall or temperature.]"
    )
    path = choose_dataset()
    if path is None:
        return
    from chap_core.cli_endpoints.validate import collect_validation_issues

    with st.spinner("Checking the data..."):
        summary = summarize_dataset(path)
        issues = collect_validation_issues(str(path))
    errors = [issue for issue in issues if issue.level == "error"]
    with st.container(border=True):
        if errors:
            st.markdown(":red[**Chap cannot use this data yet**]")
            for issue in errors[:10]:
                st.markdown(f"- {issue_line(issue)}")
        else:
            st.markdown(":green[**Chap can use this data**]")
            covariates = ", ".join(c.replace("_", " ") for c in summary.covariates) or "no covariates"
            lines = [
                (
                    f"{summary.locations} regions, {summary.period_type or 'unrecognised'} periods, "
                    f"from {summary.periods[0]} to {summary.periods[-1]}"
                ),
                f"Has {covariates}",
                "Includes a map of the regions" if summary.has_polygons else "Has no map of the regions",
            ]
            lines += [f"Note: {issue.message}" for issue in issues if issue.level == "warning"][:3]
            st.markdown("\n".join(f"- {line}" for line in lines))
        st.caption(f"Checked with `chap validate --dataset-csv {path}`.")
    guide["dataset_csv"] = str(path)
    if st.button(
        "Next: choose models",
        type="primary",
        disabled=bool(errors),
        help="Fix the problems above first." if errors else "See which models can use this data.",
    ):
        go(2)


# -- Step 2: models --


def step_models() -> None:
    st.header("Which models should compete?")
    summary = summarize_dataset(Path(guide["dataset_csv"]))
    fits: list[tuple[Any, str | None]] = []
    misfits: list[tuple[Any, str | None]] = []
    for model in marketplace.values():
        problem = model_fit(model, summary)
        (misfits if problem else fits).append((model, problem))
    if not fits:
        st.warning("None of the marketplace models can use this data. Go back and pick other data.")
    else:
        st.markdown(":gray[These models can use your data. Pick the ones to compare; Chap tests each the same way.]")
    running = running_models([model for model, _ in fits])
    # Models that already run come first: they start at once.
    fits.sort(key=lambda fit: fit[0].id not in running)
    chosen = set(guide.get("models") or [m.id for m, _ in fits[:2]])
    picked = []
    for model, _ in fits:
        with st.container(border=True):
            cols = st.columns([6, 1], vertical_alignment="top")
            status, tone = ASSESSMENT.get(model.assessed_status or "gray", ASSESSMENT["gray"])
            if cols[0].checkbox(
                f"**{model.display_name or model.id}**",
                value=model.id in chosen,
                key=f"pick:{model.id}",
                help="Include this model in the comparison.",
            ):
                picked.append(model.id)
            cols[1].badge(status, color=BADGE_COLORS[tone])
            st.markdown(model.summary or "")
            st.markdown(f":green[{fit_reason(model, summary)}]")
            if model.id in running:
                st.markdown(":blue[Running now on this machine, so it starts at once]")
    if misfits:
        with st.expander(f"{len(misfits)} models do not fit this data"):
            for model, problem in misfits:
                st.markdown(f"- **{model.display_name or model.id}** {problem}.")
    st.caption(
        "Chapkit models run as services on this machine. Chap starts the ones you pick when you run, and you "
        "choose afterwards whether to stop them."
    )
    guide["models"] = picked
    nav = st.container(horizontal=True, gap="small")
    if nav.button("Back", help="Back to your data. Your choices here are kept."):
        go(1)
    label = f"Next: {len(picked)} model{'s' if len(picked) != 1 else ''} chosen"
    if nav.button(label, type="primary", disabled=not picked, help="Pick at least one model." if not picked else None):
        go(3)


# -- Step 3: questions --


def step_questions() -> None:
    st.header("Two questions, then we test")
    st.markdown(
        ":gray[Chap pretends it is an earlier date, forecasts what came next, and compares that with what "
        "really happened.]"
    )
    summary = summarize_dataset(Path(guide["dataset_csv"]))
    unit = summary.unit
    low, high = horizon_limits([marketplace[i] for i in guide["models"] if i in marketplace])
    horizons = {n: label.replace("month", unit) for n, label in HORIZONS.items() if low <= n <= high}
    with st.container(border=True):
        horizon = st.segmented_control(
            "How far ahead do you need to forecast?",
            list(horizons),
            default=kept_choice(guide.get("horizon_n"), horizons, 3),
            # Clicking the chosen answer again would otherwise clear it and hide every thoroughness choice.
            required=bool(horizons),
            format_func=lambda n: horizons[n],
            key="guide-horizon",
            help="How many months (or weeks) ahead each test forecast reaches. Only horizons every chosen model "
            "supports are offered.",
        )
        st.caption("The time you need to act on a warning. Further ahead is harder to get right.")
    possible = {
        name: splits
        for name, splits in THOROUGHNESS.items()
        if horizon and backtest_windows(summary.periods, horizon, splits, 1)
    }
    with st.container(border=True):
        thoroughness = st.segmented_control(
            "How thorough should the test be?",
            list(possible),
            default=kept_choice(guide.get("thoroughness"), possible, "Normal"),
            required=bool(possible),
            format_func=lambda name: f"{name}: {possible[name]} tests",
            key="guide-thoroughness",
            help="How many times each model forecasts from an earlier date. More tests give a more reliable "
            "ranking and take longer.",
        )
        if thoroughness and horizon:
            windows = backtest_windows(summary.periods, horizon, possible[thoroughness], 1)
            st.caption(
                f"Each model forecasts {possible[thoroughness]} times: the first forecast starts at "
                f"{windows[0]['forecast_start']}, the last at {windows[-1]['forecast_start']}. More tests give a more "
                "reliable answer and take longer."
            )
        if len(possible) < len(THOROUGHNESS):
            st.caption("Some choices are hidden: the data is too short for them.")
    if horizon and thoroughness:
        guide.update(horizon_n=horizon, horizon=horizons[horizon], thoroughness=thoroughness)
        guide["splits"] = possible[thoroughness]
        with st.expander("Advanced: the settings these answers set"):
            st.code(format_cli_command(eval_args(guide, "<model>")), "bash", wrap_lines=True)
    nav = st.container(horizontal=True, gap="small")
    if nav.button("Back", help="Back to the models. Your answers here are kept."):
        go(2)
    if nav.button(
        "Run the comparison",
        type="primary",
        disabled=not (horizon and thoroughness),
        help="Start the models that are not running yet and test each one. This takes a few minutes.",
    ):
        guide["progress"] = {i: {"state": "waiting"} for i in guide["models"]}
        guide.setdefault("started", [])
        go(4)


# -- Step 4: run --


def running_models(models: list) -> set[str]:
    """The ids of the models with an instance on this machine, from chaps or Docker."""
    running: set[str] = set()
    if chaps_binary():
        services = {i.service_id for i in load_chaps_models(chaps_location()) if i.answering}
        running |= {m.id for m in models if m.service_id in services}
    try:
        containers = list_services({marketplace_image(m).rsplit(":", 1)[0]: m.id for m in models})
    except Exception:
        containers = []
    return running | {c.model_id for c in containers if c.status == "running" and c.url}


def step_run() -> None:
    st.header("Testing the models")
    st.markdown(":gray[You can leave this page; the tests carry on, and Start brings you back here.]")

    @st.fragment(run_every=3)
    def watch() -> None:
        # Drawn before anything is started: starting a model can take minutes, and until this list
        # replaces it the previous step would stay on screen.
        labels = {
            "waiting": ":gray[Waiting]",
            "starting": ":orange[Starting the model on this machine]",
            "evaluating": ":orange[Testing]",
            "done": ":green[Done]",
            "failed": ":red[Failed]",
        }
        for model_id, progress in guide["progress"].items():
            name = marketplace[model_id].display_name or model_id
            text = f"**{name}** · {labels[progress['state']]}"
            if progress.get("error"):
                text += f" :gray[· {progress['error']}]"
            st.markdown(text)
        # The app moves the comparison on from every page (guide_run.advance_guide); this only shows it.
        if guide["step"] == 5:
            st.rerun(scope="app")

    watch()


# -- Step 5: answer --


def step_answer() -> None:
    from chap_core.cli_endpoints.utils import compute_metrics_table

    done = {i: p for i, p in guide["progress"].items() if p["state"] == "done"}
    failed = {i: p for i, p in guide["progress"].items() if p["state"] == "failed"}
    if not done:
        st.header("No model could be tested")
        for model_id, progress in failed.items():
            st.markdown(f"- **{marketplace[model_id].display_name or model_id}**: {progress.get('error')}")
        if st.button("Back to the questions", help="Change the horizon or thoroughness and run again."):
            go(3)
        return
    files = {i: Path(p["run_dir"]) / "evaluation.nc" for i, p in done.items()}
    metrics = compute_metrics_table(list(files.values()))
    metrics["model"] = [marketplace[i].display_name or i for i in files]
    metrics["model_id"] = list(files)
    best = best_model(metrics)
    others = metrics[metrics["model_id"] != best["model_id"]]
    summary = summarize_dataset(Path(guide["dataset_csv"]))

    with st.container(border=True):
        st.caption(f"Forecasting {guide['horizon']} ahead for {guide['dataset_name']}")
        st.header(f"{best['model']} forecast best" if len(metrics) > 1 else f"{best['model']} was tested")
        st.markdown(answer_text(best, others, summary.unit))
    with st.container(border=True):
        st.markdown("**How each model did**")
        table = metrics[["model", "mae", "crps", "coverage_10_90"]].rename(
            columns={
                "model": "Model",
                "mae": "Off by, on average (cases)",
                "crps": "Overall score (lower is better)",
                "coverage_10_90": "Within its likely range",
            }
        )
        table["Within its likely range"] = (table["Within its likely range"] * 100).round().astype(int).astype(
            str
        ) + "%"
        st.dataframe(table.round(1), hide_index=True, width="stretch")
        st.caption(
            '"Off by" is the mean absolute error (MAE). The overall score is the CRPS, which also rewards a model '
            "for knowing how uncertain it is. The likely range is the 10-90% band of its forecasts."
        )
        for model_id, progress in failed.items():
            st.markdown(
                f":red[{marketplace[model_id].display_name or model_id} could not be tested:] {progress.get('error')}"
            )

    # No forecast card: chap-core has no command yet that forecasts this data with a chapkit model.
    cols = st.columns(2)
    with cols[0].container(border=True):
        st.markdown("**See the details**  \n:gray[Metrics per region, maps and forecast plots]")
        if st.button(
            "Open Results", key="guide-results", help="Metrics per region, maps and forecast plots for every model."
        ):
            st.session_state["results-selected"] = [str(f) for f in files.values()]
            guide["finished"] = True
            st.switch_page(pages["results"])
    with cols[1].container(border=True):
        started = [i for i in guide.get("started", []) if i in marketplace]
        # chaps keeps a stopped model's data; a plain Docker container is removed with it.
        kept = "their data is kept" if chaps_binary() else "stopping removes their containers"
        if started:
            st.markdown(f"**Stop the models**  \n:gray[{len(started)} started for this comparison; {kept}]")
        elif guide.get("stopped"):
            st.markdown(f"**Stopped**  \n:gray[The {guide['stopped']} started for this comparison are stopped]")
        else:
            st.markdown("**Nothing to stop**  \n:gray[The models were already running before]")
        if started and st.button(
            "Stop them", key="guide-stop", help="Stop only the models this comparison started. Others keep running."
        ):
            stop_started([marketplace[i] for i in started])
            guide.update(started=[], stopped=len(started))
            st.rerun()
    if st.button("Start over", key="guide-restart", help="Begin a new comparison. Its results stay under Runs."):
        st.session_state["guide"] = {"step": 1}
        st.rerun()


{1: step_data, 2: step_models, 3: step_questions, 4: step_run, 5: step_answer}[guide["step"]]()
