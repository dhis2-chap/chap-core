"""Streamlit entry point for `chap ui`."""

from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="CHAP", layout="wide")

pages = [
    st.Page(VIEWS / "data.py", title="1. Data", default=True),
    st.Page(VIEWS / "evaluate.py", title="2. Evaluate"),
    st.Page(VIEWS / "results.py", title="3. Results"),
]
st.navigation(pages).run()
