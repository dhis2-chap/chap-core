"""Streamlit entry point for `chap ui`."""

from pathlib import Path

import streamlit as st

from chap_core.ui.catalog import COMMAND_PAGES, CommandPage
from chap_core.ui.widgets import command_form, page_header, run_panel, sidebar_options

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="CHAP", page_icon=":material/coronavirus:", layout="wide")


def command_page(page: CommandPage):
    def render():
        page_header(page.command, page.title)
        if page.notes:
            st.info(page.notes, icon=":material/info:")
        fields, values = command_form(page.command)
        run_panel(page.command, fields, values, label=page.command)

    url = page.command.replace(" ", "-")
    return st.Page(render, title=page.title, icon=page.icon, url_path=url)


sections = {
    "Workflow": [
        st.Page(VIEWS / "data.py", title="Data", icon=":material/dataset:", default=True),
        st.Page(VIEWS / "evaluate.py", title="Evaluate", icon=":material/science:"),
        st.Page(VIEWS / "results.py", title="Results", icon=":material/insights:"),
        st.Page(VIEWS / "runs.py", title="Runs", icon=":material/history:"),
    ],
    "Models": [st.Page(VIEWS / "models.py", title="Model catalog", icon=":material/model_training:")],
}
for section, pages in COMMAND_PAGES.items():
    sections.setdefault(section, []).extend(command_page(page) for page in pages)

navigation = st.navigation(sections, expanded=True)
sidebar_options()
navigation.run()
