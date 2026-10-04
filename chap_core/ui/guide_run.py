"""Step 4 of the guide: start each chosen model, then test it.

Kept out of the guide's page so the app moves it on from whichever page is open.
"""

import time
from pathlib import Path

from chap_core.ui.models import (
    chaps_binary,
    chaps_group,
    chaps_project,
    chaps_start,
    chaps_stop,
    list_services,
    marketplace_image,
    service_info,
    start_service,
    stop_service,
)
from chap_core.ui.services import get_runs_dir, load_job, start_job
from chap_core.ui.widgets import chaps_location, load_chaps_models, load_marketplace

# How long a model may take to answer after it is started; a first start downloads its image.
START_TIMEOUT_SECONDS = 600


def eval_args(guide: dict, model_url: str) -> list[str]:
    return [
        "eval",
        "--model-name",
        model_url,
        "--dataset-csv",
        guide["dataset_csv"],
        "--output-file",
        "evaluation.nc",
        "--backtest-params.n-periods",
        str(guide["horizon_n"]),
        "--backtest-params.n-splits",
        str(guide["splits"]),
        "--backtest-params.stride",
        "1",
    ]


def running_url(model) -> str | None:
    """The address of a running instance of the model that answers, from chaps or Docker."""
    for instance in load_chaps_models(chaps_location()) if chaps_binary() else []:
        answering = instance.service_id == model.service_id and instance.answering and instance.url
        if answering and instance.url and service_info(instance.url):
            return str(instance.url)
    try:
        containers = list_services({marketplace_image(model).rsplit(":", 1)[0]: model.id})
    except Exception:
        containers = []
    for container in containers:
        if container.model_id == model.id and container.url and service_info(container.url):
            return container.url
    return None


def advance_guide(guide: dict) -> None:
    """Move every model of a running comparison a step on, and go to the answer once all are through."""
    if guide.get("step") != 4:
        return
    for model_id, progress in guide["progress"].items():
        advance(guide, model_id, progress)
    if all(p["state"] in ("done", "failed") for p in guide["progress"].values()):
        guide["step"] = 5


def advance(guide: dict, model_id: str, progress: dict) -> None:
    """Move one model a step on: start its instance, then evaluate it once it answers."""
    if progress["state"] in ("waiting", "starting"):
        model = {m.id: m for m in load_marketplace()[0]}[model_id]
        url = running_url(model)
        if url:
            name = model.display_name or model.id
            job = start_job(get_runs_dir(), eval_args(guide, url), f"Evaluate · {name}")
            progress.update(state="evaluating", url=url, run_dir=str(job.run_dir))
        elif progress["state"] == "starting" and time.time() - progress.get("started_at", 0) > START_TIMEOUT_SECONDS:
            minutes = START_TIMEOUT_SECONDS // 60
            progress.update(
                state="failed", error=f"did not answer within {minutes} minutes; its logs are in the Catalog"
            )
        elif progress["state"] == "waiting":
            try:
                if chaps_binary():
                    chaps_start(model.id, chaps_project())
                else:
                    start_service(marketplace_image(model), model.id)
                guide["started"].append(model.id)
                load_chaps_models.clear()
                progress.update(state="starting", started_at=time.time())
            except Exception as e:
                progress.update(state="failed", error=f"could not start: {e}")
    elif progress["state"] == "evaluating":
        run_dir = Path(progress["run_dir"])
        if not run_dir.exists():
            progress.update(state="failed", error="its evaluation run was deleted")
            return
        job = load_job(run_dir)
        if job.status == "succeeded":
            progress["state"] = "done"
        elif job.status in ("failed", "stopped"):
            progress.update(state="failed", error="the evaluation failed; its log is under Runs")


def stop_started(models: list) -> None:
    """Stop the instances this comparison started, keeping their data.

    The guide starts models in chap ui's own deployment or group, so instances of the same model in
    other groups are left alone.
    """
    services = {}
    if chaps_binary():
        group = chaps_group(chaps_project())
        services = {m.service_id: m for m in load_chaps_models(chaps_location()) if m.group == group}
    for model in models:
        if model.service_id in services:
            chaps_stop(services[model.service_id])
            continue
        for container in list_services({marketplace_image(model).rsplit(":", 1)[0]: model.id}):
            if container.managed:
                stop_service(container.id)
    load_chaps_models.clear()
