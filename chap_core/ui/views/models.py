"""Find models, keep your own list of them, and start chapkit models through chaps or Docker."""

from typing import Literal

import streamlit as st

from chap_core.ui.models import (
    ChapsError,
    chaps_binary,
    chaps_expose,
    chaps_project,
    chaps_start,
    chaps_stop,
    forget_model,
    list_services,
    models_file,
    remember_model,
    service_info,
    service_logs,
    start_service,
    stop_service,
)
from chap_core.ui.services import get_runs_dir
from chap_core.ui.widgets import load_catalog, load_chaps_models, model_images, page_header

BADGE_COLORS: dict[str, Literal["green", "orange", "red", "gray"]] = {
    "good": "green",
    "warn": "orange",
    "bad": "red",
    "neutral": "gray",
}

page_header(
    "Model catalog",
    "Chapkit models from the marketplace run as services, started through chaps or Docker. "
    "Other models run from a folder or a GitHub repository.",
)


def docker_services():
    try:
        return {s.model_id: s for s in list_services(model_images())}, None
    except Exception as e:
        return {}, str(e)


def use_model(model_name: str) -> None:
    st.session_state["model_name"] = model_name
    st.toast(f"Selected {model_name}", icon=":material/check:")


entries, invalid = load_catalog()
services, docker_problem = docker_services()
chaps = chaps_binary()
project = chaps_project()
deployment = load_chaps_models(str(project)) if project else {}
saved_path = models_file()

if chaps and project:
    st.caption(f"Chapkit models start in the chaps deployment at `{project}`.")
elif chaps:
    st.caption(f"Chapkit models start in a chaps deployment created at `{get_runs_dir() / 'chaps'}`.")
else:
    st.caption(
        "Chapkit models start as plain Docker containers. Install [chaps](https://github.com/winterop-com/chaps) "
        "to run them as a managed deployment instead."
    )
if docker_problem and not chaps:
    st.warning(f"Docker is not available, so chapkit models cannot be started: {docker_problem}")

with st.expander("Add a model", icon=":material/add:"):
    st.caption(f"Your models are listed in `{saved_path}`. Models you run are added automatically.")
    cols = st.columns([2, 4, 1], vertical_alignment="bottom")
    new_name = cols[0].text_input("Name", key="add-model-name")
    new_model = cols[1].text_input(
        "Model", key="add-model-value", placeholder="Folder, GitHub URL (optionally @commit) or chapkit URL"
    )
    if cols[2].button("Add", disabled=not new_model, width="stretch"):
        remember_model(saved_path, new_model, new_name or None)
        st.rerun()


def state(entry) -> tuple[str | None, str]:
    """The address a chapkit model answers on, if any, and a short description of its state."""
    in_chaps = deployment.get(entry.service_id or "")
    container = services.get(entry.id)
    url = in_chaps.url if in_chaps and in_chaps.answering else container.url if container else None
    if url and container is None and in_chaps is None:
        return None, "Not started"
    if url and service_info(url):
        return url, f":green[Running] :gray[· {url}{' · chaps' if in_chaps else ''}]"
    if in_chaps and in_chaps.internal:
        return None, ":orange[Only reachable through chap-core]"
    if in_chaps and in_chaps.state == "not-running" and container is None:
        return None, ":gray[In the chaps deployment, not running]"
    if in_chaps or container:
        return None, ":orange[Starting...]"
    return None, ":gray[Not started]"


def show_failure(message: str, error: Exception) -> None:
    st.error(f"{message}: {error}")
    if isinstance(error, ChapsError):
        with st.expander("What chaps printed"):
            st.code(error.output, "log")


def start(entry) -> None:
    with st.spinner(f"Starting {entry.name}. Pulling the image can take a while."):
        try:
            if chaps:
                chaps_start(entry.id, project)
                load_chaps_models.clear()
            else:
                start_service(entry.image, entry.id)
        except Exception as e:
            show_failure(f"Could not start {entry.name}", e)
            return
    st.rerun()


def card(entry) -> None:
    url, description = state(entry) if entry.kind == "chapkit" else (None, ":gray[Ready]")
    with st.container(border=True, key=f"card-model-{entry.id}"):
        cols = st.columns([3, 2], vertical_alignment="top")
        cols[0].markdown(f"**{entry.name}**  \n:gray[{entry.source}]")
        with cols[1]:
            st.badge(entry.status, color=BADGE_COLORS[entry.tone])
        summary = entry.summary if len(entry.summary) < 180 else entry.summary[:177].rsplit(" ", 1)[0] + "..."
        st.markdown(summary)
        st.markdown(" ".join(f":gray-badge[{tag}]" for tag in entry.tags))
        footer = st.columns([3, 2, 2], vertical_alignment="center")
        footer[0].markdown(description)
        container = services.get(entry.id)
        in_chaps = deployment.get(entry.service_id or "")
        with footer[1].popover("Details", width="stretch"):
            st.markdown(f"**{entry.name}**")
            st.markdown(entry.summary)
            if entry.image:
                st.caption(f"Image `{entry.image}`")
            if entry.model_name:
                st.caption(f"Model `{entry.model_name}`")
            if entry.repository:
                st.link_button("Source", entry.repository, icon=":material/code:")
            if entry.kind == "saved" and st.button("Remove from your models", key=f"forget:{entry.id}"):
                forget_model(saved_path, entry.model_name)
                st.rerun()
            if (
                in_chaps
                and project
                and st.button("Stop in chaps", key=f"chaps-stop:{entry.id}", icon=":material/stop:")
            ):
                try:
                    chaps_stop(entry.id, project)
                except Exception as e:
                    show_failure(f"Could not stop {entry.name}", e)
                else:
                    load_chaps_models.clear()
                    st.rerun()
            elif container and not container.managed and not in_chaps:
                st.caption(f"Running in `{container.name}`, which another tool manages.")
            elif container and container.managed and st.button("Stop", key=f"stop:{entry.id}", icon=":material/stop:"):
                stop_service(container.id)
                st.rerun()
            if container:
                st.code(service_logs(container.id, tail=60), "log", height=200)
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
            elif url:
                st.button(
                    "Use", key=f"use:{entry.id}", type="primary", width="stretch", on_click=use_model, args=(url,)
                )
            elif in_chaps and in_chaps.internal and project:
                if st.button(
                    "Expose",
                    key=f"expose:{entry.id}",
                    width="stretch",
                    help="chap eval needs to reach the model directly: chap-core's proxy only forwards reads. "
                    "This gives it a host port with `chaps models expose`.",
                ):
                    with st.spinner(f"Giving {entry.name} a port of its own..."):
                        try:
                            chaps_expose(entry.id, project)
                        except Exception as e:
                            show_failure(f"Could not expose {entry.name}", e)
                        else:
                            load_chaps_models.clear()
                            st.rerun()
            elif not (container or (in_chaps and in_chaps.state != "not-running")) and st.button(
                "Start", key=f"start:{entry.id}", width="stretch", disabled=bool(docker_problem) and not chaps
            ):
                start(entry)


running = {
    e.id
    for e in entries
    if e.kind == "chapkit"
    and (services.get(e.id) or (deployment.get(e.service_id or "") and deployment[e.service_id or ""].answering))
}
counts = {
    "All": len(entries),
    "Running": len(running),
    "Chapkit": sum(e.kind == "chapkit" for e in entries),
    "Yours": sum(e.kind == "saved" for e in entries),
    "GitHub": sum(e.kind == "github" for e in entries),
    "Local": sum(e.kind == "local" for e in entries),
}
counts = {name: n for name, n in counts.items() if n or name in ("All", "Running", "Yours")}
with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    source = st.pills(
        "Show", list(counts), default="All", format_func=lambda k: f"{k} {counts[k]}", key="catalog-source"
    )
    periods = st.pills("Period type", ["Monthly", "Weekly"], selection_mode="multi", key="catalog-period")
    query = st.text_input("Search models", placeholder="Name, covariate or author", key="catalog-search").lower()

KINDS = {"Chapkit": "chapkit", "Yours": "saved", "GitHub": "github", "Local": "local"}


def visible(entry) -> bool:
    if source == "Running" and entry.id not in running:
        return False
    if source in KINDS and entry.kind != KINDS[source]:
        return False
    if periods and not set(periods) & set(entry.tags):
        return False
    text = " ".join([entry.name, entry.summary, *entry.tags]).lower()
    return query in text


def grid() -> None:
    shown = [e for e in entries if visible(e)]
    if not shown:
        st.info("No models match." if source != "Yours" else "No models of your own yet. Add one above.")
    for first in range(0, len(shown), 3):
        for col, entry in zip(st.columns(3), shown[first : first + 3], strict=False):
            with col:
                card(entry)


starting = any(not (s.url and s.status == "running" and service_info(s.url)) for s in services.values())
starting |= any(m.state != "not-running" and not m.answering and not m.internal for m in deployment.values())
if starting:
    st.fragment(run_every=3)(grid)()
else:
    grid()
for name, error in invalid.items():
    st.caption(f"Skipped {name}: {error}")
