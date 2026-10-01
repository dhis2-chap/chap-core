"""Streamlit entry point for `chap ui`."""

from pathlib import Path

import streamlit as st

from chap_core.ui.catalog import COMMAND_PAGES, CommandPage
from chap_core.ui.widgets import command_page, context_bar, inject_css, sidebar

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="CHAP", page_icon=":material/coronavirus:", layout="wide")


def as_page(page: CommandPage):
    def render():
        command_page(page.command, page.title, page.notes)

    return st.Page(render, title=page.title, icon=page.icon, url_path=page.command.replace(" ", "-"))


sections = {
    "Workflow": [
        st.Page(VIEWS / "data.py", title="1 · Data", icon=":material/dataset:", default=True),
        st.Page(VIEWS / "evaluate.py", title="2 · Evaluate", icon=":material/science:"),
        st.Page(VIEWS / "results.py", title="3 · Results", icon=":material/insights:"),
        st.Page(VIEWS / "runs.py", title="Runs", icon=":material/history:"),
    ],
    "Models": [
        st.Page(VIEWS / "models.py", title="Catalog", icon=":material/model_training:"),
        st.Page(VIEWS / "configure.py", title="Configure a model", icon=":material/tune:"),
    ],
}
commands = {section: [as_page(page) for page in pages] for section, pages in COMMAND_PAGES.items()}

inject_css()
all_pages = [page for group in (*sections.values(), *commands.values()) for page in group]
current = st.navigation(all_pages, position="hidden")
sidebar(sections, commands, current)
top = st.container()
current.run()
# Drawn last so it shows choices the page itself just made, but placed above the page.
with top:
    context_bar()
