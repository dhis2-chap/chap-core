"""Find models, keep your own list of them, and run chapkit models as local instances through chaps or Docker."""

import time
from typing import Literal

import streamlit as st

from chap_core.ui.models import (
    CatalogEntry,
    ChapkitService,
    ChapsError,
    ChapsModel,
    chaps_binary,
    chaps_expose,
    chaps_logs,
    chaps_project,
    chaps_start,
    chaps_stop,
    chaps_test,
    forget_model,
    list_services,
    models_file,
    remember_model,
    service_info,
    service_logs,
    start_service,
    stop_service,
)
from chap_core.ui.widgets import (
    chaps_location,
    load_catalog,
    load_chaps_added_models,
    load_chaps_models,
    model_images,
    page_header,
)

BADGE_COLORS: dict[str, Literal["green", "orange", "red", "gray"]] = {
    "good": "green",
    "warn": "orange",
    "bad": "red",
    "neutral": "gray",
}

page_header(
    "Model catalog",
    "Chapkit models run as services on this machine: start one, use it in an evaluation, stop it when you "
    "are done. Other models run from a folder or a GitHub repository and need no service.",
)


def docker_services():
    try:
        return {s.model_id: s for s in list_services(model_images())}, None
    except Exception as e:
        return {}, str(e)


def show_failure(message: str, error: Exception) -> None:
    st.error(f"{message}: {error}")
    if isinstance(error, ChapsError):
        with st.expander("What chaps printed"):
            st.code(error.output, "log")


def use_model(model_name: str) -> None:
    st.session_state["model_name"] = model_name
    st.toast(f"Selected {model_name}", icon=":material/check:")


def refresh() -> None:
    load_chaps_models.clear()
    load_chaps_added_models.clear()


# How long after a start from this page a model that does not answer yet counts as starting. After
# that it is shown as not answering, and the page stops asking.
STARTUP_SECONDS = 300


def still_starting() -> bool:
    return time.time() - float(st.session_state.get("last-start", 0.0)) < STARTUP_SECONDS


def pending(model: ChapsModel) -> bool:
    """Running, but not answering yet: starting, or broken once the startup time is over."""
    return model.state not in ("not running", "registered, unreachable") and not model.answering and not model.internal


def waiting(two_lines: bool) -> str:
    gap = "  \n" if two_lines else " "
    if still_starting():
        return f":orange[Starting]{gap}:gray[{'W' if two_lines else '· w'}aiting for it to answer]"
    return f":red[Not answering]{gap}:gray[{'S' if two_lines else '· s'}ee its logs]"


def load_instances() -> bool:
    """Ask chaps and Docker what runs now. Returns whether an instance is still starting."""
    global services, docker_problem, instances, deployment, running
    services, docker_problem = docker_services()
    instances = load_chaps_models(chaps_location()) if chaps else []
    # What a catalog card shows: the model in the group chap ui starts models in, else in any other.
    deployment = {m.service_id: m for m in sorted(instances, key=lambda m: m.group == OWN_GROUP)}
    running = {
        e.id
        for e in entries
        if e.kind == "chapkit"
        and (services.get(e.id) or ((m := deployment.get(e.service_id or "")) is not None and m.answering))
    }
    starting = any(
        s.status == "running" and not (s.url and service_info(s.url)) for s in services.values() if s.managed or s.url
    )
    return still_starting() and (starting or any(pending(m) for m in instances))


entries, invalid = load_catalog()
chaps = chaps_binary()
project = chaps_project()
OWN_GROUP = None if project else "default"  # where Start puts models: the deployment, or chaps' default group
services: dict[str, ChapkitService] = {}
docker_problem: str | None = None
instances: list[ChapsModel] = []
deployment: dict[str, ChapsModel] = {}
running: set[str] = set()
starting = load_instances()
saved_path = models_file()
by_service = {e.service_id: e for e in entries if e.service_id}

if not chaps:
    st.caption(
        "Chapkit models start as plain Docker containers. Install [chaps](https://github.com/winterop-com/chaps) "
        "to manage them instead, and to start models the marketplace does not list."
    )
    with st.expander("Install chaps", icon=":material/download:"):
        st.code(
            "curl -fsSL https://raw.githubusercontent.com/winterop-com/chaps/main/install.sh | sh\nchaps doctor",
            "bash",
        )
        st.caption(
            "macOS and Linux. `chaps doctor` checks Docker and the rest. Then reload this page; restart `chap ui` "
            "if chaps went into a folder that was not on its PATH. "
            "More in the [chaps install guide](https://winterop-com.github.io/chaps/install.html)."
        )
    if docker_problem:
        st.warning(f"Docker is not available, so chapkit models cannot be started: {docker_problem}")


def start_instance(entry: CatalogEntry | None, source: str, port: int | None, everywhere: bool, model_id=None):
    """Start an instance, keeping a failure on the model's card instead of losing it on the next rerun."""
    key = f"start-failed:{entry.id if entry else source}"
    st.session_state.pop(key, None)
    st.session_state["last-start"] = time.time()
    try:
        if chaps:
            chaps_start(source, project, model_id, port, everywhere)
        elif entry is not None and entry.image:
            start_service(entry.image, entry.id)
    except Exception as e:
        st.session_state[key] = e
    refresh()
    st.rerun()


def start_dialog(entry: CatalogEntry) -> None:
    @st.dialog(f"Start {entry.name} on this machine", width="medium")
    def body() -> None:
        if chaps:
            where = f"the chaps deployment at `{project}`" if project else "chaps' own folder, not this one"
            st.markdown(
                f"- Runs as a Docker container, started with `chaps run {entry.id}` in {where}.\n"
                "- Answers on `localhost`, on a free port.\n"
                "- Keeps running after you close chap ui. Stop it under *Running on this machine*, "
                f"or with `chaps stop {entry.id}`.\n"
                "- The first start downloads the model's image, which can take a few minutes."
            )
            with st.expander("Port and network"):
                cols = st.columns(2)
                port = cols[0].number_input(
                    "Port", min_value=1024, max_value=65535, value=None, placeholder="Any free port"
                )
                reach = cols[1].selectbox("Reachable from", ["This machine only", "Other machines too"])
                if reach != "This machine only":
                    st.caption("Anyone who can reach this machine can then use the model; it has no login.")
        else:
            st.markdown(
                f"- Runs as a Docker container of `{entry.image}`.\n"
                "- Answers on `localhost`, on a free port, from this machine only.\n"
                "- Keeps running after you close chap ui, until you stop it under *Running on this machine*.\n"
                "- The first start downloads the model's image, which can take a few minutes."
            )
            port, reach = None, "This machine only"
        cols = st.columns([3, 1, 1.4])
        if cols[1].button("Cancel", width="stretch"):
            st.rerun()
        if cols[2].button("Start instance", type="primary", width="stretch"):
            with st.spinner("Starting. Downloading the image can take a while."):
                start_instance(entry, entry.id, int(port) if port else None, reach != "This machine only")

    body()


def state(entry: CatalogEntry) -> tuple[str | None, str]:
    """The address a chapkit model answers on, if any, and a short description of its state."""
    in_chaps = deployment.get(entry.service_id or "")
    container = services.get(entry.id)
    url = in_chaps.url if in_chaps and in_chaps.answering else container.url if container else None
    if url and service_info(url):
        return url, f":green[Running] :gray[· {url.removeprefix('http://')}]"
    if in_chaps and in_chaps.internal:
        return None, ":orange[Only reachable through chap-core]"
    if container and not container.managed and not container.url and in_chaps is None:
        return None, f":gray[Running in `{container.name}` without a host port]"
    if in_chaps and in_chaps.state == "not running" and container is None:
        return None, ":gray[Enabled in chaps, not running]"
    if container and container.status != "running" and in_chaps is None:
        return None, ":red[Exited]"
    if in_chaps and in_chaps.state == "registered, unreachable":
        return None, ":red[Not answering] :gray[· see its logs]"
    if in_chaps or container:
        return None, waiting(two_lines=False)
    return None, ":gray[Not running]"


# -- Running on this machine --


def instance_row(name: str, source: str, url: str | None, status: str, actions) -> None:
    cols = st.columns([2.2, 1.6, 1.6, 3.2], vertical_alignment="center")
    cols[0].markdown(f"**{name}**  \n:gray[{source}]")
    cols[1].markdown(status)
    cols[2].markdown(f"[{url.removeprefix('http://')}]({url})" if url else ":gray[-]")
    with cols[3]:
        actions()


def chaps_row(model: ChapsModel) -> None:
    entry = by_service.get(model.service_id)
    name = entry.name if entry else model.id
    answering = model.answering and model.url and service_info(model.url)
    if answering:
        status = ":green[Running]"
    elif model.internal:
        status = ":orange[Only reachable through chap-core]"
    elif model.state == "not running":
        status = ":gray[Not running]"
    elif model.state == "registered, unreachable":
        status = ":red[Not answering]  \n:gray[See its logs]"
    else:
        status = waiting(two_lines=True)
    key = f"{model.project_dir}:{model.service_id}"  # the same model may run in several groups

    def actions() -> None:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if answering:
                st.button("Use", key=f"use-run:{key}", type="primary", on_click=use_model, args=(model.url,))
            if model.internal and st.button(
                "Expose",
                key=f"expose:{key}",
                help="chap eval needs to reach the model directly: chap-core's proxy only forwards reads. "
                "This gives it a host port with `chaps models expose`.",
            ):
                with st.spinner(f"Giving {name} a port of its own..."):
                    try:
                        chaps_expose(model)
                    except Exception as e:
                        st.session_state[f"row-failed:{key}"] = (f"Could not expose {name}", e)
                    refresh()
                    st.rerun()
            if model.state == "not running" and st.button("Start", key=f"start-run:{key}", type="primary"):
                st.session_state["last-start"] = time.time()
                try:
                    chaps_start(model.id, model.project_dir)
                except Exception as e:
                    st.session_state[f"row-failed:{key}"] = (f"Could not start {name}", e)
                refresh()
                st.rerun()
            st.toggle("Logs", key=f"logs:{key}")
            if answering and st.button("Test", key=f"test:{key}"):
                with st.spinner("chaps trains and predicts with the model on generated data..."):
                    try:
                        st.session_state[f"test-result:{key}"] = chaps_test(model).strip().splitlines()[-1]
                    except Exception as e:
                        st.session_state[f"test-result:{key}"] = e
            with st.popover("Stop"):
                keep = st.button("Stop, keep its data", key=f"stop-run:{key}", width="stretch")
                st.caption(f"Starting it again picks up its configurations and trained models. `chaps stop {model.id}`")
                delete = st.button("Stop and delete its data", key=f"stop-purge:{key}", width="stretch")
                st.caption(f"Removes its data volume too. `chaps stop {model.id} --purge`")
            if keep or delete:
                try:
                    chaps_stop(model, delete_data=delete)
                except Exception as e:
                    st.session_state[f"row-failed:{key}"] = (f"Could not stop {name}", e)
                refresh()
                st.rerun()

    source = entry.source if entry else f"In `{model.project_dir}`"
    if model.group != OWN_GROUP:
        source += f" · group `{model.group}`" if model.group else f" · `{model.project_dir}`"
    instance_row(name, source, model.url, status, actions)
    row_details(key, lambda: chaps_logs(model, tail=60))


def container_row(container: ChapkitService) -> None:
    entry = next((e for e in entries if e.id == container.model_id), None)
    name = entry.name if entry else container.name
    answering = container.url and container.status == "running" and service_info(container.url)
    status = (
        ":green[Running]" if answering else ":red[Exited]" if container.status != "running" else waiting(two_lines=True)
    )
    key = container.id

    def actions() -> None:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if answering:
                st.button("Use", key=f"use-run:{key}", type="primary", on_click=use_model, args=(container.url,))
            st.toggle("Logs", key=f"logs:{key}")
            if st.button("Stop", key=f"stop-run:{key}"):
                stop_service(container.id)
                st.rerun()

    instance_row(name, f"Docker container `{container.name}`", container.url, status, actions)
    row_details(key, lambda: service_logs(container.id, tail=60))


def row_details(key: str, logs) -> None:
    """What a row's actions left behind: a failure, a test result, and its logs when asked for."""
    if failure := st.session_state.get(f"row-failed:{key}"):
        show_failure(*failure)
    result = st.session_state.get(f"test-result:{key}")
    if isinstance(result, Exception):
        show_failure("The test did not pass", result)
    elif result:
        st.success(result)
    if st.session_state.get(f"logs:{key}"):
        st.code(logs(), "log", height=220)


def running_panel() -> None:
    # Containers the UI started itself with docker run, when chaps is not installed.
    managed = [c for c in services.values() if c.managed]
    with st.container(border=True, key="card-running"):
        cols = st.columns([5, 1], vertical_alignment="top")
        cols[0].subheader("Running on this machine")
        if chaps:
            cols[0].caption(
                "Each is a Docker container started with `chaps run`. They keep running after you close chap ui, "
                "until you stop them here or with `chaps stop <id>`; `chaps ps` lists them in a terminal."
                + (f" They run in the chaps deployment at `{project}`." if project else "")
            )
        else:
            cols[0].caption(
                "Each is a Docker container, reachable from this machine only. They keep running after you close "
                "chap ui, until you stop them here."
            )
        if not instances and not managed:
            st.markdown(":gray[Nothing is running. Start a chapkit model below to use it in an evaluation.]")
            return
        # Only chap ui's own `chaps run` group: a deployment's models are its operator's to stop.
        own = [m for m in instances if m.group is not None and m.group == OWN_GROUP]
        if own and cols[1].button(
            "Stop all",
            width="stretch",
            help="Stops every model chap ui started, keeping their data. Models in other groups stay.",
        ):
            for model in own:
                try:
                    chaps_stop(model)
                except Exception as e:
                    st.session_state[f"row-failed:{model.project_dir}:{model.service_id}"] = (
                        f"Could not stop {model.id}",
                        e,
                    )
            for container in managed:
                stop_service(container.id)
            refresh()
            st.rerun()
        for model in instances:
            st.divider()
            chaps_row(model)
        for container in managed:
            st.divider()
            container_row(container)


# -- Add a model --

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
    st.divider()
    if chaps:
        st.caption(
            "Start a chapkit model the marketplace does not list, from its repository (following its newest "
            "published build) or an image. It runs as an instance like the others; starting it again reuses it."
        )
        cols = st.columns([2, 4, 1.3], vertical_alignment="bottom")
        new_id = cols[0].text_input("Id (optional)", key="add-chapkit-id")
        new_source = cols[1].text_input(
            "Chapkit model",
            key="add-chapkit-source",
            placeholder="https://github.com/org/repo, ghcr.io/org/image:tag or a local image:tag",
        )
        if cols[2].button("Start instance", key="add-chapkit", disabled=not new_source, width="stretch"):
            with st.spinner("Starting. Downloading the image can take a while."):
                start_instance(None, new_source.strip(), None, False, new_id.strip() or None)
        if failure := st.session_state.get(f"start-failed:{new_source.strip()}"):
            show_failure("Could not start the model", failure)
    else:
        st.caption(
            "With [chaps](https://github.com/winterop-com/chaps) installed, chapkit models the marketplace "
            "does not list can be started here from their repository or image."
        )


# -- The catalog --


def card(entry: CatalogEntry) -> None:
    url, description = state(entry) if entry.kind == "chapkit" else (None, ":gray[Ready]")
    failure = st.session_state.get(f"start-failed:{entry.id}")
    with st.container(border=True, key=f"card-model-{entry.id}"):
        cols = st.columns([3, 2], vertical_alignment="top")
        cols[0].markdown(f"**{entry.name}**  \n:gray[{entry.source}]")
        with cols[1]:
            st.badge(entry.status, color=BADGE_COLORS[entry.tone])
        summary = entry.summary if len(entry.summary) < 180 else entry.summary[:177].rsplit(" ", 1)[0] + "..."
        st.markdown(summary)
        st.markdown(" ".join(f":gray-badge[{tag}]" for tag in entry.tags))
        if failure:
            st.markdown(f":red[Did not start: {failure}]")
            if isinstance(failure, ChapsError):
                with st.expander("What chaps printed"):
                    st.code(failure.output, "log")
        footer = st.columns([3, 2, 2], vertical_alignment="center")
        footer[0].markdown(description)
        with footer[1].popover("Details", width="stretch"):
            st.markdown(f"**{entry.name}**")
            st.markdown(entry.summary)
            if entry.image:
                st.caption(f"Image `{entry.image}`")
            if entry.model_name:
                st.caption(f"Model `{entry.model_name}`")
            if entry.repository:
                st.link_button("Source", entry.repository, icon=":material/code:")
            if (
                entry.kind == "saved"
                and entry.model_name
                and st.button("Remove from your models", key=f"forget:{entry.id}")
            ):
                forget_model(saved_path, entry.model_name)
                st.rerun()
        with footer[2]:
            if entry.kind != "chapkit":
                st.button(
                    "Use", key=f"use:{entry.id}", type="primary", width="stretch", on_click=use_model,
                    args=(entry.model_name,),
                )  # fmt: skip
            elif url:
                st.button(
                    "Use", key=f"use:{entry.id}", type="primary", width="stretch", on_click=use_model, args=(url,)
                )
            elif not (
                ((c := services.get(entry.id)) is not None and c.status == "running")
                or ((m := deployment.get(entry.service_id or "")) is not None and m.state != "not running")
            ) and st.button(
                "Try again" if failure else "Start instance",
                key=f"start:{entry.id}",
                width="stretch",
                disabled=bool(docker_problem) and not chaps,
            ):
                start_dialog(entry)


KINDS = {"Chapkit": "chapkit", "Yours": "saved", "GitHub": "github", "Local": "local"}


def visible(entry: CatalogEntry, source, periods, query: str) -> bool:
    if source == "Running" and entry.id not in running:
        return False
    if source in KINDS and entry.kind != KINDS[source]:
        return False
    if periods and not set(periods) & set(entry.tags):
        return False
    text = " ".join([entry.name, entry.summary, *entry.tags]).lower()
    return query in text


def grid(source, periods, query: str) -> None:
    shown = [e for e in entries if visible(e, source, periods, query)]
    if not shown:
        st.info("No models match." if source != "Yours" else "No models of your own yet. Add one above.")
    for first in range(0, len(shown), 3):
        for col, entry in zip(st.columns(3), shown[first : first + 3], strict=False):
            with col:
                card(entry)


def page() -> None:
    counts = {
        "All": len(entries),
        "Running": len(running),
        "Chapkit": sum(e.kind == "chapkit" for e in entries),
        "Yours": sum(e.kind == "saved" for e in entries),
        "GitHub": sum(e.kind == "github" for e in entries),
        "Local": sum(e.kind == "local" for e in entries),
    }
    counts = {name: n for name, n in counts.items() if n or name in ("All", "Running", "Yours")}
    running_panel()
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        source = st.segmented_control(
            "Show", list(counts), default="All", format_func=lambda k: f"{k} {counts[k]}", key="catalog-source"
        )
        periods = st.segmented_control(
            "Period type", ["Monthly", "Weekly"], selection_mode="multi", key="catalog-period"
        )
        query = st.text_input("Search models", placeholder="Name, covariate or author", key="catalog-search").lower()
    grid(source, periods, query)


def watch_starting() -> None:
    """Reload the page once the instances that are starting answer. Only this empty part refreshes on
    a timer, so clicks and open menus on the page are never lost to a refresh."""
    if not load_instances():
        st.rerun()


page()
if starting:
    st.fragment(run_every=3)(watch_starting)()
for name, error in invalid.items():
    st.caption(f"Skipped {name}: {error}")
