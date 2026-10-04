"""Find the best model for my data: a guided comparison in five steps.

Data, the models that fit it, two plain questions, a run that starts what it needs, and an answer in words.
The guide's progress lives in st.session_state["guide"], so leaving the page and coming back resumes it.
"""

import time
from pathlib import Path
from typing import Any, Literal

import streamlit as st

from chap_core.ui.guide import (
    HORIZONS,
    THOROUGHNESS,
    best_model,
    fit_reason,
    horizon_limits,
    kept_choice,
    model_fit,
    summarize_dataset,
)
from chap_core.ui.models import (
    ASSESSMENT,
    chaps_binary,
    chaps_project,
    chaps_start,
    chaps_stop,
    list_services,
    marketplace_image,
    service_info,
    start_service,
    stop_service,
)
from chap_core.ui.services import (
    PUBLISHED_DATASETS,
    backtest_windows,
    fetch_published_dataset,
    get_runs_dir,
    get_uploads_dir,
    load_job,
    save_upload,
    start_job,
)
from chap_core.ui.widgets import chaps_location, load_chaps_models, load_marketplace

STEPS = ["Your data", "Models", "Questions", "Run", "Answer"]
# How long a model may take to answer after it is started; a first start downloads its image.
START_TIMEOUT_SECONDS = 600
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
    source = st.segmented_control("Where is your data?", sources, default=sources[0], key="guide-source")
    if source == "Use an example":
        name = st.selectbox("Example", list(PUBLISHED_DATASETS), key="guide-example")
        st.caption("Published in dhis2/climate-health-data, with a map of the regions.")
        with st.spinner("Downloading the dataset..."):
            path = fetch_published_dataset(uploads_dir, PUBLISHED_DATASETS[name])
        guide["dataset_name"] = name
        return path
    if source == "Upload my own file":
        cols = st.columns(2)
        csv_file = cols[0].file_uploader("Your data, as a CSV file", type="csv", key="guide-upload")
        geojson = cols[1].file_uploader("A map of the regions (optional GeoJSON)", type=["geojson", "json"])
        if csv_file is None:
            return None
        path = save_upload(uploads_dir, csv_file.name, csv_file.getvalue())
        if geojson is not None:
            save_upload(uploads_dir, path.with_suffix(".geojson").name, geojson.getvalue())
        guide["dataset_name"] = path.name
        return path
    choice = st.selectbox("Earlier file", earlier, format_func=lambda p: p.name, key="guide-earlier")
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
                st.markdown(f"- {issue.message}" + (f" ({issue.location})" if issue.location else ""))
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
    if st.button("Next: choose models", type="primary", disabled=bool(errors)):
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
    chosen = set(guide.get("models") or [m.id for m, _ in fits[:2]])
    picked = []
    for model, _ in fits:
        with st.container(border=True):
            cols = st.columns([6, 1], vertical_alignment="top")
            status, tone = ASSESSMENT.get(model.assessed_status or "gray", ASSESSMENT["gray"])
            if cols[0].checkbox(
                f"**{model.display_name or model.id}**", value=model.id in chosen, key=f"pick:{model.id}"
            ):
                picked.append(model.id)
            cols[1].badge(status, color=BADGE_COLORS[tone])
            st.markdown(model.summary or "")
            st.markdown(f":green[{fit_reason(model, summary)}]")
    if misfits:
        with st.expander(f"{len(misfits)} models do not fit this data"):
            for model, problem in misfits:
                st.markdown(f"- **{model.display_name or model.id}** {problem}.")
    st.caption(
        "Chapkit models run as services on this machine. Chap starts the ones you pick when you run, and you "
        "choose afterwards whether to stop them."
    )
    guide["models"] = picked
    cols = st.columns([1, 1, 4])
    if cols[0].button("Back"):
        go(1)
    label = f"Next: {len(picked)} model{'s' if len(picked) != 1 else ''} chosen"
    if cols[1].button(label, type="primary", disabled=not picked):
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
            format_func=lambda n: horizons[n],
            key="guide-horizon",
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
            format_func=lambda name: f"{name}: {possible[name]} tests",
            key="guide-thoroughness",
        )
        if thoroughness and horizon:
            windows = backtest_windows(summary.periods, horizon, possible[thoroughness], 1)
            st.caption(
                f"Each model forecasts {possible[thoroughness]} times, starting from {windows[0]['forecast_start']} "
                f"to {windows[-1]['forecast_start']}. More tests give a more reliable answer and take longer."
            )
        if len(possible) < len(THOROUGHNESS):
            st.caption("Some choices are hidden: the data is too short for them.")
    if horizon and thoroughness:
        guide.update(horizon_n=horizon, horizon=horizons[horizon], thoroughness=thoroughness)
        guide["splits"] = possible[thoroughness]
        with st.expander("Advanced: the settings these answers set"):
            st.code(eval_args("<model>")[1:], "bash", wrap_lines=True)
    cols = st.columns([1, 1, 4])
    if cols[0].button("Back"):
        go(2)
    if cols[1].button("Run the comparison", type="primary", disabled=not (horizon and thoroughness)):
        guide["progress"] = {i: {"state": "waiting"} for i in guide["models"]}
        guide.setdefault("started", [])
        go(4)


def eval_args(model_url: str) -> list[str]:
    return [
        "eval",
        "--model-name",
        model_url,
        "--dataset-csv",
        guide["dataset_csv"],
        "--output-file",
        "evaluation.nc",
        "--backtest-params.n-periods",
        str(guide["horizon_n"]),
        "--backtest-params.n-splits",
        str(guide["splits"]),
        "--backtest-params.stride",
        "1",
    ]


# -- Step 4: run --


def running_url(model) -> str | None:
    """The address of a running instance of the model that answers, from chaps or Docker."""
    for instance in load_chaps_models(chaps_location()) if chaps_binary() else []:
        answering = instance.service_id == model.service_id and instance.answering and instance.url
        if answering and instance.url and service_info(instance.url):
            return str(instance.url)
    try:
        containers = list_services({marketplace_image(model).rsplit(":", 1)[0]: model.id})
    except Exception:
        containers = []
    for container in containers:
        if container.model_id == model.id and container.url and service_info(container.url):
            return container.url
    return None


def advance(model_id: str, progress: dict) -> None:
    """Move one model a step on: start its instance, then evaluate it once it answers."""
    model = marketplace[model_id]
    if progress["state"] in ("waiting", "starting"):
        url = running_url(model)
        if url:
            name = model.display_name or model.id
            job = start_job(get_runs_dir(), eval_args(url), f"Evaluate · {name}")
            progress.update(state="evaluating", url=url, run_dir=str(job.run_dir))
        elif progress["state"] == "starting" and time.time() - progress.get("started_at", 0) > START_TIMEOUT_SECONDS:
            minutes = START_TIMEOUT_SECONDS // 60
            progress.update(
                state="failed", error=f"did not answer within {minutes} minutes; its logs are in the Catalog"
            )
        elif progress["state"] == "waiting":
            try:
                if chaps_binary():
                    chaps_start(model.id, chaps_project())
                else:
                    start_service(marketplace_image(model), model.id)
                guide["started"].append(model.id)
                load_chaps_models.clear()
                progress.update(state="starting", started_at=time.time())
            except Exception as e:
                progress.update(state="failed", error=f"could not start: {e}")
    elif progress["state"] == "evaluating":
        job = load_job(Path(progress["run_dir"]))
        if job.status == "succeeded":
            progress["state"] = "done"
        elif job.status in ("failed", "stopped"):
            progress.update(state="failed", error="the evaluation failed; its log is under Runs")


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
        if all(p["state"] in ("done", "failed") for p in guide["progress"].values()):
            guide["step"] = 5
            st.rerun(scope="app")
        for model_id, progress in guide["progress"].items():
            advance(model_id, progress)

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
        if st.button("Back to the questions"):
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
        text = f"Its forecasts were off by {best['mae']:.0f} cases per region and {summary.unit} on average"
        if len(others):
            text += ", against " + ", ".join(f"{row.mae:.0f} for {row.model}" for row in others.itertuples())
        text += (
            f". The real number of cases fell inside its likely range {best['coverage_10_90'] * 100:.0f}% of the "
            "time; a model that knows how uncertain it is gets close to 80%."
        )
        st.markdown(text)
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

    cols = st.columns(3)
    with cols[0].container(border=True):
        st.markdown(f"**Forecast the coming months**  \n:gray[With {best['model']}, on all your data]")
        if st.button("Forecast", type="primary", key="guide-forecast"):
            st.session_state["model_name"] = done[best["model_id"]]["url"]
            st.session_state["dataset_csv"] = guide["dataset_csv"]
            guide["finished"] = True
            st.switch_page(pages["forecast"])
    with cols[1].container(border=True):
        st.markdown("**See the details**  \n:gray[Metrics per region, maps and forecast plots]")
        if st.button("Open Results", key="guide-results"):
            st.session_state["results-selected"] = [str(f) for f in files.values()]
            guide["finished"] = True
            st.switch_page(pages["results"])
    with cols[2].container(border=True):
        started = [i for i in guide.get("started", []) if i in marketplace]
        st.markdown(
            f"**Stop the models**  \n:gray[{len(started)} started for this comparison; their data is kept]"
            if started
            else "**Nothing to stop**  \n:gray[The models were already running before]"
        )
        if started and st.button("Stop them", key="guide-stop"):
            stop_started(started)
            guide["started"] = []
            st.rerun()
    if st.button("Start over", key="guide-restart"):
        st.session_state["guide"] = {"step": 1}
        st.rerun()


def stop_started(model_ids: list[str]) -> None:
    """Stop the instances this comparison started, keeping their data."""
    services = {m.service_id: m for m in load_chaps_models(chaps_location())} if chaps_binary() else {}
    for model_id in model_ids:
        model = marketplace[model_id]
        if model.service_id in services:
            chaps_stop(services[model.service_id])
            continue
        for container in list_services({marketplace_image(model).rsplit(":", 1)[0]: model.id}):
            if container.managed:
                stop_service(container.id)
    load_chaps_models.clear()


{1: step_data, 2: step_models, 3: step_questions, 4: step_run, 5: step_answer}[guide["step"]]()
