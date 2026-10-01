"""Find models, run chapkit models locally and write model configurations."""

import re

import pandas as pd
import streamlit as st
import yaml

from chap_core.ui.models import (
    github_models,
    list_services,
    marketplace_image,
    service_info,
    service_logs,
    start_service,
    stop_service,
)
from chap_core.ui.services import example_models, get_workdir, run_job

STATUS_COLORS = {"green": "green", "yellow": "orange", "orange": "orange", "red": "red", "gray": "gray"}

st.title("Model catalog")
st.caption(
    "Pick a model to evaluate. Chapkit models from the marketplace run as local Docker containers; "
    "other models run from a local directory or a GitHub repository."
)


def use_model(model_name: str) -> None:
    st.session_state["model_name"] = model_name
    st.toast(f"Selected model `{model_name}`", icon=":material/check:")


@st.cache_data(ttl=600, show_spinner="Loading the model marketplace...")
def load_marketplace():
    from chap_core.services.model_marketplace import list_models

    models, invalid = list_models()
    return [m.model_dump(mode="json") for m in models], invalid, {m.id: marketplace_image(m) for m in models}


def docker_problem() -> str | None:
    try:
        list_services()
    except Exception as e:
        return str(e)
    return None


marketplace_tab, services_tab, other_tab, config_tab = st.tabs(
    ["Marketplace", "Running services", "GitHub and local", "Configure"]
)

with marketplace_tab:
    try:
        entries, invalid, images = load_marketplace()
    except Exception as e:
        st.error(f"Could not load the marketplace: {e}")
        entries, invalid, images = [], {}, {}
    problem = docker_problem()
    if problem:
        st.warning(f"Docker is not available, so chapkit models cannot be started: {problem}")
    kinds = st.segmented_control("Show", ["model", "template"], selection_mode="multi", default=["model"])
    running = {s.model_id: s for s in ([] if problem else list_services())}
    for entry in (e for e in entries if e["kind"] in (kinds or [])):
        with st.container(border=True):
            cols = st.columns([4, 1], vertical_alignment="center")
            status = entry.get("assessed_status") or "gray"
            cols[0].markdown(
                f"### {entry.get('display_name') or entry['id']} "
                f":{STATUS_COLORS[status]}-badge[{status}] :gray-badge[{entry['kind']}]"
            )
            cols[0].caption(entry.get("summary") or "")
            compat, covs, attribution = entry["compatibility"], entry["covariates"], entry["attribution"]
            details = [
                f"Period: {', '.join(compat.get('period_types') or ['any'])}",
                f"Horizon: {compat.get('min_prediction_periods')}-{compat.get('max_prediction_periods')}",
                f"Covariates: {', '.join(covs.get('required') or []) or 'none'}",
                f"By: {attribution.get('author') or '-'}"
                + (f" ({attribution['organization']})" if attribution.get("organization") else ""),
            ]
            cols[0].markdown(" · ".join(details))
            cols[0].caption(f"`{images[entry['id']]}`")
            with cols[1]:
                service = running.get(entry["id"])
                if service is not None:
                    st.button(
                        "Use",
                        key=f"use:{entry['id']}",
                        icon=":material/check:",
                        width="stretch",
                        on_click=use_model,
                        args=(service.url,),
                        disabled=service.url is None,
                    )
                    st.caption(f"Running at {service.url}")
                elif st.button(
                    "Start",
                    key=f"start:{entry['id']}",
                    icon=":material/play_arrow:",
                    width="stretch",
                    disabled=bool(problem),
                ):
                    with st.spinner(f"Starting {images[entry['id']]} (pulling the image can take a while)..."):
                        try:
                            start_service(images[entry["id"]], entry["id"])
                        except Exception as e:
                            st.error(str(e))
                    st.rerun()
                if entry["source"].get("repository"):
                    st.link_button("Source", entry["source"]["repository"], icon=":material/code:", width="stretch")
    for name, error in invalid.items():
        st.caption(f"Skipped {name}: {error}")

with services_tab:
    services = [] if docker_problem() else list_services()
    if not services:
        st.info("No chapkit services are running. Start one from the Marketplace tab.")

    @st.fragment(run_every=3)
    def show_services():
        for service in list_services():
            with st.container(border=True):
                info = service_info(service.url) if service.url and service.status == "running" else None
                ready = info is not None
                icon = ":material/check_circle:" if ready else ":material/hourglass_top:"
                st.markdown(
                    f"{icon} **{service.model_id}** · `{service.url}` · {service.status}{'' if ready else ' (starting)'}"
                )
                st.caption(f"`{service.image}`")
                cols = st.columns(3)
                cols[0].button(
                    "Use for evaluation",
                    key=f"use-svc:{service.id}",
                    icon=":material/check:",
                    on_click=use_model,
                    args=(service.url,),
                    disabled=not ready,
                )
                if cols[1].button("Stop", key=f"stop-svc:{service.id}", icon=":material/stop:"):
                    stop_service(service.id)
                    st.rerun()
                if info:
                    with st.expander("Service info"):
                        st.json(info)
                with st.expander("Container log"):
                    st.code(service_logs(service.id), "log", height=250)

    if services:
        show_services()

with other_tab:
    rows = [{"Name": p.name, "Source": "Example", "Model name": str(p)} for p in example_models()]
    rows += [{"Name": m.name, "Source": "GitHub", "Model name": m.model_name} for m in github_models()]
    table = pd.DataFrame(rows)
    selection = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="other-models")
    if selection.selection.rows:
        chosen = table.iloc[selection.selection.rows[0]]["Model name"]
        st.button("Use selected model", icon=":material/check:", on_click=use_model, args=(chosen,))
    custom = st.text_input("Or enter a model directory, GitHub URL or chapkit URL")
    if custom:
        st.button("Use this model", icon=":material/check:", on_click=use_model, args=(custom,), key="use-custom")

with config_tab:
    st.markdown(
        "Load the options a model accepts (`chap model schema`), fill them in, and save a "
        "configuration file to use with **Evaluate** and the other commands."
    )
    model_name = st.text_input("Model", value=st.session_state.get("model_name", ""), key="config-model")
    if st.button("Load options", icon=":material/download:", disabled=not model_name):
        with st.spinner("Loading model options..."):
            job = run_job(
                get_workdir(),
                ["model", "schema", "--model-name", model_name, "--output-file", "schema.yaml"],
                "model schema",
            )
        schema_file = job.run_dir / "schema.yaml"
        if job.status == "succeeded" and schema_file.exists():
            st.session_state["config-schema"] = (model_name, yaml.safe_load(schema_file.read_text()))
        else:
            st.error("Could not load the model's options.")
            st.code(job.log.read_text(errors="replace")[-5000:], "log")

    loaded = st.session_state.get("config-schema")
    if loaded and loaded[0] == model_name:
        schema = loaded[1]["properties"]
        options = schema["user_option_values"]["properties"]
        values = {}
        if not options:
            st.info("This model has no options.")
        for name, spec in options.items():
            label, help_text, default = (
                name.replace("_", " ").capitalize(),
                spec.get("description"),
                spec.get("default"),
            )
            kind = spec.get("type")
            if "enum" in spec:
                values[name] = st.selectbox(
                    label,
                    spec["enum"],
                    index=spec["enum"].index(default) if default in spec["enum"] else 0,
                    help=help_text,
                )
            elif kind == "boolean":
                values[name] = st.checkbox(label, value=bool(default), help=help_text)
            elif kind == "integer":
                values[name] = st.number_input(label, value=default, step=1, help=help_text)
            elif kind == "number":
                values[name] = st.number_input(label, value=None if default is None else float(default), help=help_text)
            elif kind == "array":
                text = st.text_area(
                    label, "\n".join(map(str, default or [])), help=(help_text or "") + " One per line."
                )
                values[name] = [line for line in text.splitlines() if line.strip()]
            else:
                values[name] = st.text_input(label, value=default or "", help=help_text)
        covariates_spec = schema["additional_continuous_covariates"]
        covariates = []
        if covariates_spec.get("maxItems") != 0:
            text = st.text_area("Additional continuous covariates", help="Extra covariate columns, one per line.")
            covariates = [line.strip() for line in text.splitlines() if line.strip()]
        config = {"user_option_values": values, "additional_continuous_covariates": covariates}
        st.code(yaml.safe_dump(config, sort_keys=False), "yaml")
        default_name = re.sub(r"[^A-Za-z0-9]+", "-", model_name.split("@")[0].rstrip("/").split("/")[-1]).strip("-")
        file_name = st.text_input("File name", value=f"{default_name}.yaml")
        if st.button("Save configuration", type="primary", icon=":material/save:"):
            path = get_workdir() / "configs" / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            st.session_state["model_configuration_yaml"] = str(path)
            st.success(f"Saved `{path}`. It is now selected as the model configuration.")
