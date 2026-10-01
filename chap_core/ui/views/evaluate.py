"""Configure and run a backtest evaluation (`chap eval`)."""

import streamlit as st

from chap_core.ui.models import model_label
from chap_core.ui.services import EXAMPLE_MODEL
from chap_core.ui.widgets import command_form, page_header, run_panel

page_header("eval", "Evaluate a model")
st.markdown(
    "Trains the model on expanding windows of the dataset and forecasts the following periods, "
    "so predictions can be compared with what actually happened."
)

basic = {
    "model_name",
    "dataset_csv",
    "backtest_params.n_periods",
    "backtest_params.n_splits",
    "backtest_params.stride",
    "model_configuration_yaml",
}
prefill = (
    {"model_name": str(EXAMPLE_MODEL)} if EXAMPLE_MODEL.exists() and not st.session_state.get("model_name") else {}
)
fields, values = command_form("eval", basic=basic, prefill=prefill)

run_panel("eval", fields, values, label=f"eval {model_label(values.get('model_name') or 'model')}")
