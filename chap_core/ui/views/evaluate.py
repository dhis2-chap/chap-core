"""Configure and run a backtest evaluation (`chap eval`)."""

import streamlit as st
from pydantic import ValidationError

from chap_core.api_types import BacktestParams
from chap_core.ui.services import (
    EVAL_OUTPUT_NAME,
    EXAMPLE_MODEL,
    LOG_NAME,
    build_eval_command,
    format_cli_command,
    get_workdir,
    new_run_dir,
    start_eval,
)

workdir = get_workdir()

st.title("Evaluate a model")

dataset_csv = st.session_state.get("dataset_csv")
if dataset_csv is None:
    st.warning("Pick a dataset first.")
    st.page_link("views/data.py", label="Go to Data", icon=":material/arrow_back:")
    st.stop()
st.markdown(f"Dataset: `{dataset_csv}`")

model_name = st.text_input(
    "Model",
    value=str(EXAMPLE_MODEL) if EXAMPLE_MODEL.exists() else "",
    help="Local model directory, GitHub URL (e.g. https://github.com/dhis2-chap/minimalist_example_r) "
    "or the URL of a running chapkit service.",
)

st.subheader("Backtest")
cols = st.columns(3)
n_periods = cols[0].number_input("Periods ahead to forecast", min_value=1, value=3)
n_splits = cols[1].number_input("Number of test splits", min_value=1, value=7)
stride = cols[2].number_input("Periods between splits", min_value=1, value=1)
config_file = st.file_uploader("Model configuration YAML (optional)", type=["yaml", "yml"])

try:
    backtest_params = BacktestParams(n_periods=n_periods, n_splits=n_splits, stride=stride)
except ValidationError as e:
    st.error(str(e))
    st.stop()

preview_output = workdir / "runs" / "<run>" / EVAL_OUTPUT_NAME
preview_config = workdir / "runs" / "<run>" / "model_config.yaml" if config_file else None
st.caption("Equivalent CLI command")
st.code(
    format_cli_command(build_eval_command(model_name, dataset_csv, preview_output, backtest_params, preview_config)),
    "bash",
)

run = st.session_state.get("eval_run")
running = run is not None and run["proc"].poll() is None

if st.button("Run evaluation", type="primary", disabled=running or not model_name):
    run_dir = new_run_dir(workdir, model_name)
    config_path = None
    if config_file is not None:
        config_path = run_dir / "model_config.yaml"
        config_path.write_bytes(config_file.getvalue())
    args = build_eval_command(model_name, dataset_csv, run_dir / EVAL_OUTPUT_NAME, backtest_params, config_path)
    st.session_state["eval_run"] = {"proc": start_eval(args, run_dir), "run_dir": run_dir}
    st.rerun()


def show_run():
    run = st.session_state["eval_run"]
    proc, run_dir = run["proc"], run["run_dir"]
    returncode = proc.poll()
    if returncode is None:
        st.info(f"Running in `{run_dir}` ...")
        if st.button("Stop"):
            proc.terminate()
    elif returncode == 0 and (run_dir / EVAL_OUTPUT_NAME).exists():
        st.success(f"Finished. Results written to `{run_dir / EVAL_OUTPUT_NAME}`")
        st.session_state["selected_evals"] = [str(run_dir / EVAL_OUTPUT_NAME)]
        st.page_link("views/results.py", label="See results", icon=":material/arrow_forward:")
    else:
        st.error(f"The evaluation failed (exit code {returncode}). See the log below.")
    log_lines = (run_dir / LOG_NAME).read_text(errors="replace").splitlines()
    st.code("\n".join(log_lines[-200:]) or "(no output yet)", "log", height=400)
    if returncode is not None and running:
        st.rerun()


if run is not None:
    st.subheader("Run")
    if running:
        st.fragment(run_every=2)(show_run)()
    else:
        show_run()
