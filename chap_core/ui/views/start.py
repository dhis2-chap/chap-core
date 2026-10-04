"""Where chap ui opens: pick a goal, and get walked through it."""

import streamlit as st

from chap_core.ui.widgets import page_header

page_header(
    "What do you want to do?",
    "Pick a goal and Chap walks you through it step by step. Every step shows the command it runs, "
    "so you can repeat it in a terminal later.",
)

pages = st.session_state["pages"]
guide = st.session_state.get("guide")
if guide and guide.get("step", 1) > 1 and not guide.get("finished"):
    with st.container(border=True, key="card-continue"):
        cols = st.columns([5, 1], vertical_alignment="center")
        cols[0].markdown(
            f":gray[Continue where you left off]  \n**Find the best model for {guide['dataset_name']}** · "
            f"step {guide['step']} of 5"
        )
        if cols[1].button("Continue", type="primary", width="stretch", help="Go back to the comparison you started."):
            st.switch_page(pages["guide"])


def use_case(key: str, title: str, text: str, steps: str, target: str, highlight: bool = False) -> None:
    with st.container(border=True, key=f"card-usecase-{key}"):
        st.markdown(f"#### {title}")
        st.markdown(text)
        st.caption(steps)
        if st.button(
            "Start" if highlight else "Open",
            key=f"usecase:{key}",
            type="primary" if highlight else "secondary",
            help=steps,
        ):
            if key == "compare":
                st.session_state["guide"] = {"step": 1}
            st.switch_page(pages[target])


cols = st.columns(2)
with cols[0]:
    use_case(
        "compare",
        "Find the best model for my data",
        "Test several models on your own data, the way a forecast would have gone in the past, "
        "and see which predicts best.",
        "Your data, models that fit it, two questions, then the answer.",
        "guide",
        highlight=True,
    )
    use_case(
        "check",
        "Check a model I built",
        "Point Chap at your model's folder, GitHub repository or chapkit service. It checks that the model "
        "loads, trains and predicts.",
        "Opens the sanity check; a guided version is coming.",
        "sanity-check-model",
    )
with cols[1]:
    use_case(
        "forecast",
        "Forecast the coming months",
        "Train a model on all your data and forecast ahead, with the uncertainty of each forecast.",
        "Opens the forecast command; a guided version is coming.",
        "forecast",
    )
    use_case(
        "explore",
        "Look at a dataset",
        "See what a dataset holds, map its regions, and find out whether Chap can use it as it is.",
        "Opens the Data page.",
        "data",
    )

st.caption(
    "Looking at earlier results? Open **Results** in the sidebar. Know the command you want? Press Ctrl+K (⌘K on a Mac)."
)
