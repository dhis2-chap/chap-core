"""Streamlit building blocks shared by the chap UI pages."""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from chap_core.ui.catalog import command_title
from chap_core.ui.commands import build_args, command_fields, command_help
from chap_core.ui.services import (
    command_name,
    example_models,
    format_cli_command,
    get_runs_dir,
    get_uploads_dir,
    job_outputs,
    list_evaluations,
    list_jobs,
    load_job,
    option_value,
    resolve_paths,
    save_upload,
    start_job,
    stop_job,
    workspace_files,
)

if TYPE_CHECKING:
    from chap_core.ui.commands import Field
    from chap_core.ui.services import Job

# Run-config options that are set once in the sidebar instead of on every form.
GLOBAL_RUN_OPTIONS = {
    "run_config.debug": "Debug logging",
    "run_config.ignore_environment": "Skip environment setup",
    "run_config.track": "Track with MLflow",
}

GROUP_TITLES = {
    "backtest-params": "More backtest options",
    "run-config": "Model run",
    "estimator-options": "Hyperparameter search",
    "lime-params": "Explanation method",
    "learning-params": "Learning",
}

STATUS = {
    "running": ":blue[:material/progress_activity:] Running",
    "succeeded": ":green[:material/check_circle:] Succeeded",
    "failed": ":red[:material/error:] Failed",
    "stopped": ":gray[:material/stop_circle:] Stopped",
}

PATH_SUFFIXES = {
    "dataset_csv": (".csv",),
    "counterfactual_csv": (".csv",),
    "data_filename": (".csv",),
    "dataset_path": (".csv",),
    "geojson_file": (".geojson", ".json"),
    "geojson_path": (".geojson", ".json"),
    "polygons_json": (".geojson", ".json"),
    "model_configuration_yaml": (".yaml", ".yml"),
    "model_config_path": (".yaml", ".yml"),
    "search_space_yaml": (".yaml", ".yml"),
    "estimator_options.search_space": (".yaml", ".yml"),
    "data_source_mapping": (".json",),
    "request_json": (".json",),
    "input_file": (".nc",),
    "input_files": (".nc",),
    "evaluation_path": (".nc",),
}

# Values that follow the user across pages once chosen.
SHARED_VALUES = {
    "dataset_csv": "dataset_csv",
    "model_name": "model_name",
    "model_url": "model_name",
    "model_configuration_yaml": "model_configuration_yaml",
    "input_file": "evaluation",
    "evaluation_path": "evaluation",
}

CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-left: 3rem; padding-right: 3rem; max-width: none; }
[class*="st-key-card"] { background: #FFFFFF; }
.st-key-context-bar { border-bottom: 1px solid #E3E6EA; padding-bottom: 0.6rem; margin-bottom: 0.4rem; }
.st-key-context-bar button { border-radius: 6px; min-height: 2rem; padding: 0.1rem 0.85rem; }
[data-testid="stColumn"]:has([class*="st-key-card-run"]) { position: sticky; top: 1rem; align-self: flex-start; z-index: 2; }
.chap-brand { display: flex; flex-direction: column; padding: 0 0.5rem 0.25rem; line-height: 1.3; }
.chap-brand b { font-size: 1.15rem; letter-spacing: 0.02em; }
.chap-brand span, .chap-muted { color: #5B6470; font-size: 0.8rem; }
.chap-nav-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: #5B6470; margin: 0.9rem 0 0.15rem 0.5rem; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def settings() -> dict[str, Any]:
    """Global options chosen in the sidebar."""
    current: dict[str, Any] = st.session_state.setdefault("settings", dict.fromkeys(GLOBAL_RUN_OPTIONS, False))
    return current


def sidebar(sections: dict[str, list], commands: dict[str, list], current) -> None:
    """Navigation: the main sections as links, the remaining commands searchable and grouped, then run options."""
    running = sum(job.status == "running" for job in list_jobs(get_runs_dir()))
    with st.sidebar:
        st.html('<div class="chap-brand"><b>CHAP</b><span>Modeling workbench</span></div>')
        for section, pages in sections.items():
            st.html(f'<div class="chap-nav-label">{section}</div>')
            for page in pages:
                label = f"{page.title} · {running} running" if page.title == "Runs" and running else None
                st.page_link(page, label=label)
        st.html('<div class="chap-nav-label">More commands</div>')
        query = st.text_input(
            "Find a command", placeholder="Find a command...", label_visibility="collapsed", key="nav-search"
        ).lower()
        for group, pages in commands.items():
            matches = [page for page in pages if query in page.title.lower()]
            if matches:
                is_current = current.url_path in {page.url_path for page in pages}
                with st.expander(f"{group} · {len(matches)}", expanded=bool(query) or is_current):
                    for page in matches:
                        st.page_link(page)
        st.html('<div class="chap-nav-label">Run options</div>')
        options = settings()
        for key, label in GLOBAL_RUN_OPTIONS.items():
            options[key] = st.toggle(label, value=options[key], key=f"settings:{key}")
        st.caption(f"Runs folder `{get_runs_dir()}`")


def context_bar() -> None:
    """What the user is working with, shown on top of every page; each chip leads to where it is changed."""
    dataset = st.session_state.get("dataset_csv")
    model = st.session_state.get("model_name")
    config = st.session_state.get("model_configuration_yaml")
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="context-bar"):
        st.markdown(":gray[Working with]", width="content")
        chips = [
            ("dataset", f":gray[Dataset] {short_name(dataset)}" if dataset else "+ Dataset", "views/data.py"),
            ("model", f":gray[Model] {_model_name(model)}" if model else "+ Model", "views/models.py"),
            (
                "config",
                f":gray[Configuration] {Path(config).name}" if config else "+ Model configuration",
                "views/configure.py",
            ),
        ]
        for key, label, target in chips:
            if st.button(label, key=f"chip-{key}", type="secondary" if label[0] != "+" else "tertiary"):
                st.switch_page(target)
        if config:
            st.button(
                "",
                icon=":material/close:",
                key="chip-config-clear",
                type="tertiary",
                help="Stop using this model configuration",
                on_click=clear_shared,
                args=("model_configuration_yaml",),
            )


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.markdown(f":gray[{description}]")


def short_name(value: str) -> str:
    """A file name for paths, the value itself for URLs."""
    return value if "://" in value else Path(value).name


def time_ago(moment: datetime) -> str:
    seconds = int((datetime.now() - moment).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return moment.strftime("%Y-%m-%d")


def duration(job: Job) -> str:
    seconds = int(((job.finished or datetime.now()) - job.started).total_seconds())
    return f"{seconds // 60}:{seconds % 60:02d}"


def job_title(job: Job) -> str:
    """What a job did, in words: the command's page title and the model it ran."""
    model = option_value(job.args, "--model-name") or option_value(job.args, "--model-url")
    command = command_name(job.args)
    title = command_title(command)
    if not model:
        return title
    name = _model_name(model)
    if model.startswith("http") and name == model.rstrip("/").split("/")[-1]:
        # A chapkit service that is no longer running: its model is recorded in the run directory name.
        name = job.name.split("_", 1)[-1].removeprefix(command.replace(" ", "-") + "-") or name
    return f"{title} · {name}"


def command_form(
    command: str,
    *,
    key: str | None = None,
    basic: set[str] | None = None,
    exclude: frozenset[str] = frozenset(),
    prefill: dict[str, Any] | None = None,
    title: str = "Inputs",
) -> tuple[list[Field], dict[str, Any]]:
    """Render a form for a chap command and return its fields and the entered values.

    Fields listed in `basic` (by default the required ones) go in a card; the others are grouped
    into collapsed sections below it. The global run options come from the sidebar.
    """
    key = key or command
    fields = [f for f in command_fields(command) if f.key not in exclude]
    prefill = prefill or {}
    values: dict[str, Any] = {f.key: settings()[f.key] for f in fields if f.key in GLOBAL_RUN_OPTIONS}
    shown = [f for f in fields if f.key not in GLOBAL_RUN_OPTIONS]

    def is_basic(field: Field) -> bool:
        return field.key in basic if basic is not None else field.required

    main = [f for f in shown if is_basic(f)]
    if main:
        with st.container(border=True, key=f"card-{key}-inputs"):
            st.subheader(title)
            values |= field_rows(main, key, prefill)
    values |= advanced_sections([f for f in shown if not is_basic(f)], key, prefill)
    return fields, values


def field_rows(fields: list[Field], form_key: str, prefill: dict[str, Any]) -> dict[str, Any]:
    """Render fields, putting runs of small inputs side by side."""
    values = {}
    compact = ("int", "float", "bool", "choice")
    i = 0
    while i < len(fields):
        run = [fields[i]]
        while fields[i].kind in compact and i + len(run) < len(fields) and fields[i + len(run)].kind in compact:
            run.append(fields[i + len(run)])
        if len(run) == 1:
            values[run[0].key] = field_widget(run[0], form_key, prefill)
        else:
            for col, field in zip(st.columns(len(run)), run, strict=True):
                with col:
                    values[field.key] = field_widget(field, form_key, prefill)
        i += len(run)
    return values


def advanced_sections(fields: list[Field], form_key: str, prefill: dict[str, Any]) -> dict[str, Any]:
    """One collapsed section per option group, plus one for the remaining options."""
    groups: dict[str, list[Field]] = {}
    for field in fields:
        title = GROUP_TITLES.get(field.group or "", "More options")
        groups.setdefault(title, []).append(field)
    values = {}
    for title, group_fields in sorted(groups.items(), key=lambda item: item[0] == "More options"):
        with st.expander(title):
            cols = st.columns(2)
            for i, field in enumerate(group_fields):
                with cols[i % 2]:
                    values[field.key] = field_widget(field, form_key, prefill)
    return values


def run_panel(
    command: str,
    fields: list[Field],
    values: dict[str, Any],
    label: str,
    *,
    key: str | None = None,
    summary: dict[str, str] | None = None,
    run_label: str = "Run",
) -> Job | None:
    """The card with what will run, the run button, the CLI command and the state of the latest run."""
    key = key or command
    missing = [f.label for f in fields if f.required and values.get(f.key) in (None, "", [])]
    args = build_args(command, fields, resolve_paths(fields, values, Path.cwd()))
    job_key = f"job:{key}"
    current = load_job(st.session_state[job_key]) if job_key in st.session_state else None
    running = current is not None and current.status == "running"
    with st.container(border=True, key=f"card-run-{key}"):
        st.subheader("Ready to run" if not missing else "Missing input")
        if missing:
            st.markdown(f":gray[Fill in {', '.join(missing)}.]")
        if summary:
            st.markdown("  \n".join(f":gray[{name}] &nbsp; {value}" for name, value in summary.items()))
        if st.button(
            run_label,
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=bool(missing) or running,
            key=f"run:{key}",
        ):
            model = values.get("model_name") or values.get("model_url")
            if model:
                from chap_core.ui.models import models_file, remember_model

                remember_model(models_file(), model)
            st.session_state[job_key] = start_job(get_runs_dir(), args, label).run_dir
            st.rerun()
        with st.expander("Show as CLI command"):
            st.code(format_cli_command(args), "bash", wrap_lines=True)
        if current is not None:
            job_status(current)
    return current


def job_status(job: Job) -> None:
    """One line with a job's state, refreshing while it runs."""
    if job.status == "running":
        st.fragment(run_every=2)(_job_status_body)(job.run_dir)
    else:
        _job_status_body(job.run_dir)


def _job_status_body(run_dir: Path) -> None:
    job = load_job(run_dir)
    cols = st.columns([3, 2], vertical_alignment="center")
    cols[0].markdown(f"{STATUS[job.status]} :gray[{duration(job)}]")
    if job.status == "running":
        if cols[1].button("Stop", icon=":material/stop:", key=f"stop:{job.name}", width="stretch"):
            stop_job(job)
            st.rerun()
    else:
        evaluations = [p for p in job_outputs(job) if p.suffix == ".nc"]
        if evaluations and cols[1].button(
            "Results", icon=":material/insights:", key=f"res:{job.name}", width="stretch"
        ):
            open_in_results(evaluations[0])


def job_details(job: Job) -> None:
    """Log and outputs of a job, full width. Refreshes while the job runs."""
    if job.status == "running":
        st.fragment(run_every=2)(_job_details_body)(job.run_dir, True)
    else:
        _job_details_body(job.run_dir, False)


def _job_details_body(run_dir: Path, was_running: bool) -> None:
    job = load_job(run_dir)
    if was_running and job.status != "running":
        st.rerun()
    log = job.log.read_text(errors="replace").splitlines() if job.log.exists() else []
    with st.expander("Log", expanded=job.status in ("running", "failed")):
        st.code("\n".join(log[-300:]) or "(no output yet)", "log", height=350)
    if job.status != "running":
        show_outputs(job)


def recent_runs(limit: int = 3) -> None:
    """The latest runs, as a short list with a link to all of them."""
    jobs = list_jobs(get_runs_dir())[:limit]
    with st.container(border=True, key="card-recent"):
        cols = st.columns([3, 2], vertical_alignment="center")
        cols[0].subheader("Recent runs")
        if cols[1].button("All runs", type="tertiary", key="recent-all"):
            st.switch_page("views/runs.py")
        if not jobs:
            st.markdown(":gray[Nothing has run yet.]")
        for job in jobs:
            st.markdown(f"{STATUS[job.status].split(' ')[0]} {job_title(job)} :gray[· {time_ago(job.started)}]")


def command_page(command: str, title: str, notes: str = "") -> None:
    """A generic page for one chap command: form on the left, run panel on the right, run details below."""
    page_header(title, command_help(command))
    if notes:
        st.info(notes, icon=":material/info:")
    left, right = st.columns([2, 1], gap="large")
    with left:
        fields, values = command_form(command)
    with right:
        model = values.get("model_name") or values.get("model_url")
        label = f"{command} {_model_name(model)}" if model else command
        job = run_panel(command, fields, values, label=label, run_label=f"Run {title.lower()}")
    if job is not None:
        job_details(job)


def open_in_results(path: Path) -> None:
    st.session_state["evaluation"] = str(path)
    st.session_state["selected_evals"] = [str(path)]
    st.session_state["results-selected"] = [str(path)]
    st.switch_page("views/results.py")


def _model_name(model: str | None) -> str:
    from chap_core.ui.models import model_label

    return model_label(model) if model else ""


def show_outputs(job: Job) -> None:
    """Render every file a job produced according to its type."""
    outputs = job_outputs(job)
    if not outputs:
        return
    st.subheader("Outputs")
    for path in outputs:
        rel = path.relative_to(job.run_dir)
        with st.expander(str(rel), expanded=len(outputs) <= 3 or path.suffix in (".html", ".md")):
            _show_file(path)
            st.download_button("Download", path.read_bytes(), path.name, key=f"dl:{path}", icon=":material/download:")


def _show_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".nc":
        st.caption("Evaluation file")
        if st.button("Open in Results", key=f"open:{path}", icon=":material/insights:"):
            open_in_results(path)
    elif suffix == ".html":
        st.iframe(path, height="content")
    elif suffix in (".png", ".jpg", ".jpeg", ".svg"):
        st.image(str(path))
    elif suffix == ".csv":
        st.dataframe(pd.read_csv(path), hide_index=True)
    elif suffix == ".md":
        st.markdown(path.read_text(errors="replace"))
    elif suffix in (".json", ".yaml", ".yml", ".txt", ".log", ".geojson"):
        text = path.read_text(errors="replace")
        st.code(text[:20000], "json" if suffix in (".json", ".geojson") else "yaml")
    elif suffix == ".pdf":
        st.caption("PDF document")
    else:
        st.caption(f"{path.stat().st_size} bytes")


def _initial(field: Field, prefill: dict[str, Any]) -> Any:
    if field.key in prefill:
        return prefill[field.key]
    shared = SHARED_VALUES.get(field.key)
    if shared and st.session_state.get(shared):
        return st.session_state[shared]
    if shared == "evaluation":
        latest = list_evaluations(get_runs_dir())
        if latest:
            return str(latest[0])
    return field.default


def field_widget(field: Field, form_key: str, prefill: dict[str, Any]) -> Any:
    widget_key = f"{form_key}:{field.key}"
    initial = _initial(field, prefill)
    label = field.label
    help_text = field.help or None
    if field.kind == "bool":
        return st.checkbox(label, value=bool(initial), help=help_text, key=widget_key)
    if field.kind == "int":
        return st.number_input(
            label,
            value=initial,
            step=1,
            help=help_text,
            key=widget_key,
            placeholder="default",
            min_value=None if field.minimum is None else int(field.minimum),
            max_value=None if field.maximum is None else int(field.maximum),
        )
    if field.kind == "float":
        return st.number_input(
            label,
            value=initial,
            help=help_text,
            key=widget_key,
            placeholder="default",
            min_value=field.minimum,
            max_value=field.maximum,
        )
    if field.kind == "choice":
        options = list(field.choices)
        index = options.index(str(initial)) if initial is not None and str(initial) in options else None
        return st.selectbox(label, options, index=index, help=help_text, key=widget_key, placeholder="default")
    if field.kind == "multichoice":
        return st.multiselect(label, list(field.choices), default=initial or [], help=help_text, key=widget_key)
    if field.kind == "list":
        return _list_widget(field, widget_key, initial, help_text)
    if field.key in ("model_name", "model_url"):
        return _model_widget(field, widget_key, initial, help_text)
    if field.key in PATH_SUFFIXES or field.kind == "path":
        return _path_widget(field, widget_key, initial, help_text)
    return st.text_input(label, value=initial or "", help=help_text, key=widget_key) or None


def _sync_shared(field: Field, widget_key: str, initial) -> None:
    """Start a text widget from its initial value, and follow the shared value when another page changes it."""
    shared = SHARED_VALUES.get(field.key)
    current = st.session_state.get(shared) if shared else None
    synced_key = f"{widget_key}:synced"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = str(initial) if initial else ""
    elif shared in st.session_state and current != st.session_state.get(synced_key):
        st.session_state[widget_key] = current or ""
    st.session_state[synced_key] = current


def _publish_shared(field: Field, widget_key: str, value: str) -> None:
    """Make a value typed into a form, or its removal, the shared choice for the other pages."""
    shared = SHARED_VALUES.get(field.key)
    if shared and (value or shared in st.session_state):
        st.session_state[shared] = value or None
        st.session_state[f"{widget_key}:synced"] = value or None


def clear_shared(name: str) -> None:
    """Forget a shared choice, such as the model configuration, on every page."""
    st.session_state[name] = None


def _list_widget(field: Field, widget_key: str, initial, help_text) -> list[str]:
    suffixes = PATH_SUFFIXES.get(field.key)
    if suffixes:
        files = [str(p) for p in workspace_files(get_runs_dir(), get_uploads_dir(), suffixes)]
        default = [v for v in (initial or st.session_state.get("selected_evals") or []) if v in files]
        return st.multiselect(field.label, files, default=default, help=help_text, key=widget_key, format_func=_short)
    text = st.text_area(
        field.label,
        value="\n".join(map(str, initial or [])),
        help=(help_text or "") + " One value per line.",
        key=widget_key,
        height=100,
    )
    return [line.strip() for line in text.splitlines() if line.strip()]


def _path_widget(field: Field, widget_key: str, initial, help_text) -> str | None:
    suffixes = PATH_SUFFIXES.get(field.key)
    _sync_shared(field, widget_key, initial)
    cols = st.columns([5, 1], vertical_alignment="bottom")
    value = cols[0].text_input(field.label, help=help_text, key=widget_key)
    _publish_shared(field, widget_key, value)
    if not field.is_output:
        with cols[1].popover("Browse", icon=":material/folder_open:", width="stretch"):
            files = workspace_files(get_runs_dir(), get_uploads_dir(), suffixes) if suffixes else []
            if files:
                st.selectbox(
                    "Workspace files",
                    files,
                    index=None,
                    format_func=_short,
                    key=f"{widget_key}:pick",
                    on_change=_copy_choice,
                    args=(f"{widget_key}:pick", widget_key),
                )
            st.file_uploader(
                "Upload",
                type=[s.lstrip(".") for s in suffixes] if suffixes else None,
                key=f"{widget_key}:upload",
                on_change=_store_upload,
                args=(f"{widget_key}:upload", widget_key),
            )
            if value:
                st.button(
                    "Clear",
                    icon=":material/close:",
                    key=f"{widget_key}:clear",
                    on_click=_clear_widget,
                    args=(widget_key,),
                )
    return value or None


def _clear_widget(widget_key: str) -> None:
    st.session_state[widget_key] = ""


def _model_widget(field: Field, widget_key: str, initial, help_text) -> str | None:
    from chap_core.ui.models import github_models

    _sync_shared(field, widget_key, initial)
    cols = st.columns([5, 1], vertical_alignment="bottom")
    value = cols[0].text_input(
        field.label, help=help_text, key=widget_key, placeholder="Directory, GitHub URL or chapkit URL"
    )
    _publish_shared(field, widget_key, value)
    with cols[1].popover("Choose", icon=":material/model_training:", width="stretch"):
        from chap_core.ui.models import models_file, saved_models

        options = _running_service_options()
        options |= {f"Yours: {m.name}": m.model for m in saved_models(models_file())}
        options |= {f"Example: {p.name}": str(p) for p in example_models()}
        options |= {f"GitHub: {m.name}": m.model_name for m in github_models()}
        st.selectbox(
            "Known models",
            list(options),
            index=None,
            key=f"{widget_key}:pick",
            on_change=_copy_choice,
            args=(f"{widget_key}:pick", widget_key, options),
        )
        if st.button("Browse all models", key=f"{widget_key}:browse", icon=":material/arrow_forward:"):
            st.switch_page("views/models.py")
    return value or None


@st.cache_data(ttl=600, show_spinner="Loading the model marketplace...")
def load_marketplace():
    """The marketplace's models and why any entry could not be read; refreshed every ten minutes."""
    from chap_core.services.model_marketplace import list_models

    try:
        return list_models()
    except Exception as e:
        return [], {"marketplace": str(e)}


def load_catalog():
    """Every model the UI knows: marketplace, models chaps started from a URL or image, your saved
    models and a checkout's models."""
    from chap_core.ui.models import catalog_entries, models_file, saved_models

    marketplace, invalid = load_marketplace()
    entries = catalog_entries(marketplace, saved_models(models_file()))
    entries[len(marketplace) : len(marketplace)] = load_chaps_added_models(chaps_location())
    return entries, invalid


def chaps_location() -> str:
    """Cache key for where chaps models live: the deployment's folder, or "" for `chaps run` groups."""
    from chap_core.ui.models import chaps_project

    project = chaps_project()
    return str(project) if project else ""


@st.cache_data(ttl=60, show_spinner=False)
def load_chaps_added_models(location: str):
    """Running models chaps started from a URL or an image."""
    from chap_core.ui.models import chaps_added_models

    return chaps_added_models(load_chaps_models(location))


@st.cache_data(ttl=5, show_spinner=False)
def load_chaps_models(location: str):
    """The models chaps runs; asked again at most every few seconds."""
    from chap_core.ui.models import chaps_models

    return chaps_models(Path(location) if location else None)


def model_images() -> dict[str, str]:
    """Image repository of each chapkit model in the catalog, mapped to its catalog id."""
    entries, _ = load_catalog()
    return {e.image.rsplit(":", 1)[0]: e.id for e in entries if e.image}


def _running_service_options() -> dict[str, str]:
    """Chapkit services that are up: containers of marketplace images, and a chaps deployment's models."""
    from chap_core.ui.models import list_services, model_label

    urls: list[str] = []
    with contextlib.suppress(Exception):  # Docker may not be available
        urls += [s.url for s in list_services(model_images()) if s.url and s.status == "running"]
    urls += [m.url for m in load_chaps_models(chaps_location()) if m.answering and m.url]
    return {f"Running: {model_label(url)}": url for url in dict.fromkeys(urls)}


def _copy_choice(source_key: str, target_key: str, mapping: dict[str, str] | None = None) -> None:
    choice = st.session_state.get(source_key)
    if choice is not None:
        st.session_state[target_key] = mapping[choice] if mapping else str(choice)


def _store_upload(source_key: str, target_key: str) -> None:
    upload = st.session_state.get(source_key)
    if upload is not None:
        st.session_state[target_key] = str(save_upload(get_uploads_dir(), upload.name, upload.getvalue()))


def _short(value) -> str:
    path = Path(value)
    runs = get_runs_dir()
    return str(path.relative_to(runs)) if path.is_relative_to(runs) else path.name
