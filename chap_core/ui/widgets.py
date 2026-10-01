"""Streamlit building blocks shared by the chap UI pages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from chap_core.ui.commands import build_args, command_fields, command_help
from chap_core.ui.services import (
    example_models,
    format_cli_command,
    get_workdir,
    job_outputs,
    load_job,
    resolve_paths,
    save_upload,
    start_job,
    stop_job,
    workspace_files,
)

if TYPE_CHECKING:
    from chap_core.ui.commands import Field
    from chap_core.ui.services import Job

# Run-config options that can be set once in the sidebar instead of on every form.
GLOBAL_RUN_OPTIONS = {
    "run_config.debug": "Debug logging",
    "run_config.ignore_environment": "Skip environment setup",
    "run_config.track": "Track runs with MLflow",
}

STATUS_ICONS = {"running": ":material/progress_activity:", "succeeded": ":material/check_circle:"}
STATUS_ICONS |= {"failed": ":material/error:", "stopped": ":material/stop_circle:"}

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


def settings() -> dict[str, Any]:
    """Global options chosen in the sidebar."""
    current: dict[str, Any] = st.session_state.setdefault(
        "settings", {"show_cli": True} | dict.fromkeys(GLOBAL_RUN_OPTIONS, False)
    )
    return current


def sidebar_options() -> None:
    """Global run options and workspace info, shown under the navigation."""
    current = settings()
    with st.sidebar:
        st.markdown("**Run options**")
        for key, label in GLOBAL_RUN_OPTIONS.items():
            current[key] = st.toggle(label, value=current[key], key=f"settings:{key}")
        current["show_cli"] = st.toggle("Show CLI commands", value=current["show_cli"], key="settings:show_cli")
        st.divider()
        for name, label in (("dataset_csv", "Dataset"), ("model_name", "Model"), ("evaluation", "Evaluation")):
            value = st.session_state.get(name)
            st.caption(f"{label}: `{Path(value).name if value and '://' not in value else value or '-'}`")
        st.caption(f"Workspace: `{get_workdir()}`")


def command_form(
    command: str,
    *,
    key: str | None = None,
    basic: set[str] | None = None,
    exclude: frozenset[str] = frozenset(),
    prefill: dict[str, Any] | None = None,
) -> tuple[list[Field], dict[str, Any]]:
    """Render a form for a chap command and return its fields and the entered values.

    Fields listed in `basic` are shown directly; everything else goes under "Advanced options".
    Without `basic`, required fields are shown directly.
    """
    key = key or command
    fields = [f for f in command_fields(command) if f.key not in exclude]
    prefill = prefill or {}
    values: dict[str, Any] = {}

    def is_basic(field: Field) -> bool:
        return field.key in basic if basic is not None else field.required

    for field in fields:
        if field.key in GLOBAL_RUN_OPTIONS:
            values[field.key] = settings()[field.key]
    fields_to_show = [f for f in fields if f.key not in GLOBAL_RUN_OPTIONS]
    main = [f for f in fields_to_show if is_basic(f)]
    advanced = [f for f in fields_to_show if not is_basic(f)]
    compact = ("int", "float", "bool", "choice")
    i = 0
    while i < len(main):
        # Put runs of small inputs side by side; text and file inputs get a full row.
        run = [main[i]]
        while main[i].kind in compact and i + len(run) < len(main) and main[i + len(run)].kind in compact:
            run.append(main[i + len(run)])
        if len(run) == 1:
            values[run[0].key] = _field_widget(run[0], key, prefill)
        else:
            for col, field in zip(st.columns(len(run)), run, strict=True):
                with col:
                    values[field.key] = _field_widget(field, key, prefill)
        i += len(run)
    if advanced:
        with st.expander("Advanced options"):
            groups: dict[str | None, list[Field]] = {}
            for field in advanced:
                groups.setdefault(field.group, []).append(field)
            for group, group_fields in groups.items():
                if group:
                    st.markdown(f"**{group.replace('-', ' ').capitalize()}**")
                cols = st.columns(2)
                for i, field in enumerate(group_fields):
                    with cols[i % 2]:
                        values[field.key] = _field_widget(field, key, prefill)
    return fields, values


def run_panel(command: str, fields: list[Field], values: dict[str, Any], label: str, key: str | None = None) -> None:
    """Show the equivalent CLI command, a run button, and the latest job for this form."""
    key = key or command
    missing = [f.label for f in fields if f.required and values.get(f.key) in (None, "", [])]
    args = build_args(command, fields, resolve_paths(fields, values, Path.cwd()))
    if settings()["show_cli"]:
        st.code(format_cli_command(args), "bash", wrap_lines=True)
    if missing:
        st.caption(f"Required: {', '.join(missing)}")
    job_key = f"job:{key}"
    current = load_job(st.session_state[job_key]) if job_key in st.session_state else None
    running = current is not None and current.status == "running"
    if st.button(
        "Run", type="primary", icon=":material/play_arrow:", disabled=bool(missing) or running, key=f"run:{key}"
    ):
        st.session_state[job_key] = start_job(get_workdir(), args, label).run_dir
        st.rerun()
    if current is not None:
        job_panel(current)


def job_panel(job: Job) -> None:
    """Status, live log and outputs of a job. Refreshes itself while the job runs."""
    if job.status == "running":
        st.fragment(run_every=2)(_job_body)(job.run_dir, True)
    else:
        _job_body(job.run_dir, False)


def _job_body(run_dir: Path, was_running: bool) -> None:
    job = load_job(run_dir)
    if was_running and job.status != "running":
        st.rerun()
    cols = st.columns([3, 1])
    duration = ((job.finished or pd.Timestamp.now().to_pydatetime()) - job.started).seconds
    cols[0].markdown(f"{STATUS_ICONS[job.status]} **{job.status.capitalize()}** after {duration}s · `{job.name}`")
    if job.status == "running" and cols[1].button("Stop", icon=":material/stop:", key=f"stop:{job.name}"):
        stop_job(job)
        st.rerun()
    log = job.log.read_text(errors="replace").splitlines() if job.log.exists() else []
    with st.expander("Log", expanded=job.status != "succeeded"):
        st.code("\n".join(log[-300:]) or "(no output yet)", "log", height=350)
    if job.status != "running":
        show_outputs(job)


def show_outputs(job: Job) -> None:
    """Render every file a job produced according to its type."""
    outputs = job_outputs(job)
    if not outputs:
        return
    st.markdown("**Outputs**")
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
            st.session_state["evaluation"] = str(path)
            st.session_state["selected_evals"] = [str(path)]
            st.switch_page("views/results.py")
    elif suffix == ".html":
        components.html(path.read_text(errors="replace"), height=600, scrolling=True)
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
    return field.default


def _field_widget(field: Field, form_key: str, prefill: dict[str, Any]) -> Any:
    widget_key = f"{form_key}:{field.key}"
    initial = _initial(field, prefill)
    label = field.label
    help_text = field.help or None
    if field.kind == "bool":
        return st.checkbox(label, value=bool(initial), help=help_text, key=widget_key)
    if field.kind == "int":
        return st.number_input(label, value=initial, step=1, help=help_text, key=widget_key, placeholder="default")
    if field.kind == "float":
        return st.number_input(label, value=initial, help=help_text, key=widget_key, placeholder="default")
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
    elif current and current != st.session_state.get(synced_key):
        st.session_state[widget_key] = current
    st.session_state[synced_key] = current


def _publish_shared(field: Field, widget_key: str, value: str) -> None:
    """Make a value typed into a form the shared choice for the other pages."""
    shared = SHARED_VALUES.get(field.key)
    if shared and value:
        st.session_state[shared] = value
        st.session_state[f"{widget_key}:synced"] = value


def _list_widget(field: Field, widget_key: str, initial, help_text) -> list[str]:
    suffixes = PATH_SUFFIXES.get(field.key)
    if suffixes:
        files = [str(p) for p in workspace_files(get_workdir(), suffixes)]
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
            files = workspace_files(get_workdir(), suffixes) if suffixes else []
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
    return value or None


def _model_widget(field: Field, widget_key: str, initial, help_text) -> str | None:
    from chap_core.ui.models import github_models

    _sync_shared(field, widget_key, initial)
    cols = st.columns([5, 1], vertical_alignment="bottom")
    value = cols[0].text_input(
        field.label, help=help_text, key=widget_key, placeholder="Directory, GitHub URL or chapkit URL"
    )
    _publish_shared(field, widget_key, value)
    with cols[1].popover("Choose", icon=":material/model_training:", width="stretch"):
        options = {**_running_service_options(), **{f"Example: {p.name}": str(p) for p in example_models()}}
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


def _running_service_options() -> dict[str, str]:
    try:
        from chap_core.ui.models import list_services

        return {f"Chapkit: {s.model_id}": s.url for s in list_services() if s.url and s.status == "running"}
    except Exception:
        return {}


def _copy_choice(source_key: str, target_key: str, mapping: dict[str, str] | None = None) -> None:
    choice = st.session_state.get(source_key)
    if choice is not None:
        st.session_state[target_key] = mapping[choice] if mapping else str(choice)


def _store_upload(source_key: str, target_key: str) -> None:
    upload = st.session_state.get(source_key)
    if upload is not None:
        st.session_state[target_key] = str(save_upload(get_workdir(), upload.name, upload.getvalue()))


def _short(value) -> str:
    path = Path(value)
    runs = get_workdir() / "runs"
    return str(path.relative_to(runs)) if path.is_relative_to(runs) else path.name


def page_header(command: str, title: str) -> None:
    st.title(title)
    st.caption(command_help(command))
