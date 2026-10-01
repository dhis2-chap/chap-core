"""Find models and run chapkit models locally."""

from typing import Literal

import streamlit as st

from chap_core.ui.models import (
    catalog_entries,
    list_services,
    service_info,
    service_logs,
    start_service,
    stop_service,
)
from chap_core.ui.widgets import page_header

BADGE_COLORS: dict[str, Literal["green", "orange", "red", "gray"]] = {
    "good": "green",
    "warn": "orange",
    "bad": "red",
    "neutral": "gray",
}

page_header(
    "Model catalog",
    "Chapkit models run as local Docker containers. Other models run from a folder or a GitHub repository.",
)


@st.cache_data(ttl=600, show_spinner="Loading the model marketplace...")
def load_catalog():
    from chap_core.services.model_marketplace import list_models

    try:
        marketplace, invalid = list_models()
    except Exception as e:
        marketplace, invalid = [], {"marketplace": str(e)}
    return catalog_entries(marketplace), invalid


def docker_services():
    try:
        return {s.model_id: s for s in list_services()}, None
    except Exception as e:
        return {}, str(e)


def use_model(model_name: str) -> None:
    st.session_state["model_name"] = model_name
    st.toast(f"Selected {model_name}", icon=":material/check:")


entries, invalid = load_catalog()
services, docker_problem = docker_services()
if docker_problem:
    st.warning(f"Docker is not available, so chapkit models cannot be started: {docker_problem}")

counts = {
    "All": len(entries),
    "Running": len(services),
    "Chapkit": sum(e.kind == "chapkit" for e in entries),
    "GitHub": sum(e.kind == "github" for e in entries),
    "Local": sum(e.kind == "local" for e in entries),
}
with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    source = st.pills(
        "Show", list(counts), default="All", format_func=lambda k: f"{k} {counts[k]}", key="catalog-source"
    )
    periods = st.pills("Period type", ["Monthly", "Weekly"], selection_mode="multi", key="catalog-period")
    query = st.text_input("Search models", placeholder="Name, covariate or author", key="catalog-search").lower()


def visible(entry) -> bool:
    if source == "Running" and entry.id not in services:
        return False
    if source in ("Chapkit", "GitHub", "Local") and entry.kind != source.lower():
        return False
    if periods and not set(periods) & set(entry.tags):
        return False
    text = " ".join([entry.name, entry.summary, *entry.tags]).lower()
    return query in text


def card(entry) -> None:
    service = services.get(entry.id)
    ready = bool(service and service.url and service.status == "running" and service_info(service.url))
    with st.container(border=True, key=f"card-model-{entry.id}"):
        cols = st.columns([3, 2], vertical_alignment="top")
        cols[0].markdown(f"**{entry.name}**  \n:gray[{entry.source}]")
        with cols[1]:
            st.badge(entry.status, color=BADGE_COLORS[entry.tone])
        summary = entry.summary if len(entry.summary) < 180 else entry.summary[:177].rsplit(" ", 1)[0] + "..."
        st.markdown(summary)
        st.markdown(" ".join(f":gray-badge[{tag}]" for tag in entry.tags))
        footer = st.columns([3, 2, 2], vertical_alignment="center")
        if service:
            footer[0].markdown(f":green[Running] :gray[· {service.url}]" if ready else ":orange[Starting...]")
        else:
            footer[0].markdown(":gray[Not started]" if entry.kind == "chapkit" else ":gray[Ready]")
        with footer[1].popover("Details", width="stretch"):
            st.markdown(f"**{entry.name}**")
            st.markdown(entry.summary)
            if entry.image:
                st.caption(f"Image `{entry.image}`")
            if entry.model_name:
                st.caption(f"Model `{entry.model_name}`")
            if entry.repository:
                st.link_button("Source", entry.repository, icon=":material/code:")
            if service:
                if st.button("Stop service", key=f"stop:{entry.id}", icon=":material/stop:"):
                    stop_service(service.id)
                    st.rerun()
                st.code(service_logs(service.id, tail=60), "log", height=200)
        with footer[2]:
            if entry.kind != "chapkit":
                st.button(
                    "Use",
                    key=f"use:{entry.id}",
                    type="primary",
                    width="stretch",
                    on_click=use_model,
                    args=(entry.model_name,),
                )
            elif service:
                st.button(
                    "Use",
                    key=f"use:{entry.id}",
                    type="primary",
                    width="stretch",
                    on_click=use_model,
                    args=(service.url,),
                    disabled=not ready,
                )
            elif st.button("Start", key=f"start:{entry.id}", width="stretch", disabled=bool(docker_problem)):
                with st.spinner(f"Starting {entry.name}. Pulling the image can take a while."):
                    try:
                        start_service(entry.image, entry.id)
                    except Exception as e:
                        st.error(str(e))
                st.rerun()


def grid() -> None:
    shown = [e for e in entries if visible(e)]
    if not shown:
        st.info("No models match.")
    for start in range(0, len(shown), 3):
        for col, entry in zip(st.columns(3), shown[start : start + 3], strict=False):
            with col:
                card(entry)


starting = [s for s in services.values() if not (s.url and s.status == "running" and service_info(s.url))]
if starting:
    st.fragment(run_every=3)(grid)()
else:
    grid()
for name, error in invalid.items():
    st.caption(f"Skipped {name}: {error}")
