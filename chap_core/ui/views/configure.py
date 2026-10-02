"""Write a model configuration file from the options the model declares (`chap model schema`)."""

import re

import streamlit as st
import yaml

from chap_core.ui.services import get_runs_dir, run_job
from chap_core.ui.widgets import clear_shared, page_header

page_header(
    "Configure a model",
    "Load the options a model accepts, fill them in, and save a configuration file "
    "that Evaluate and the other commands use.",
)

current = st.session_state.get("model_configuration_yaml")
if current:
    with st.container(border=True, key="card-configure-current"):
        cols = st.columns([4, 1], vertical_alignment="center")
        cols[0].markdown(
            f"**In use:** `{current}`  \n:gray[Evaluate and the other commands pass this file to the model.]"
        )
        cols[1].button(
            "Stop using it",
            icon=":material/close:",
            width="stretch",
            on_click=clear_shared,
            args=("model_configuration_yaml",),
        )
        with st.expander("Show file"):
            try:
                st.code(open(current).read(), "yaml")
            except OSError as e:
                st.warning(str(e))
else:
    st.info("No model configuration is in use; models run with their defaults.", icon=":material/info:")

with st.container(border=True, key="card-configure"):
    model_name = st.text_input("Model", value=st.session_state.get("model_name", ""), key="config-model")
    if st.button("Load options", icon=":material/download:", disabled=not model_name):
        with st.spinner("Loading model options..."):
            job = run_job(
                get_runs_dir(),
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
            path = get_runs_dir() / "configs" / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            st.session_state["model_configuration_yaml"] = str(path)
            st.success(f"Saved `{path}`. It is now selected as the model configuration.")
