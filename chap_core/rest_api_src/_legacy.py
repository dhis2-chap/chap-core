import json
from typing import Union

from fastapi import UploadFile, BackgroundTasks

from chap_core.api_types import RequestV1
from chap_core.rest_api import internal_state, worker
from chap_core.rest_api_src import worker_functions as wf


async def set_model_path(model_path: str) -> dict:
    """
    Set the model to be used for training and evaluation
    https://github.com/knutdrand/external_rmodel_example.git
    """
    internal_state.model_path = model_path
    return {"status": "success"}


async def post_zip_file(
    file: Union[UploadFile, None] = None, background_tasks: BackgroundTasks = None
) -> dict:
    """
    Post a zip file containing the data needed for training and evaluation, and start the training
    """

    # Herman: I comment these two lines out, since we accept more than one jobs for now?
    # if not internal_state.is_ready():
    #    raise HTTPException(status_code=400, detail="Model is currently training")

    model_name, model_path = "HierarchicalModel", None
    if internal_state.model_path is not None:
        model_name = "external"
        model_path = internal_state.model_path

    job = worker.queue(wf.train_on_zip_file, file, model_name, model_path)
    internal_state.current_job = job

    return {"status": "success"}


async def predict_from_json(data: RequestV1) -> dict:
    """
    Post a json file containing the data needed for prediction
    """
    model_path = "https://github.com/sandvelab/chap_auto_ewars"
    if internal_state.model_path is not None:
        model_name = "external"
        model_path = internal_state.model_path
    json_data = data.model_dump()

    str_data = json.dumps(json_data)

    job = worker.queue(wf.train_on_json_data, str_data, model_path, model_path)
    internal_state.current_job = job

    return {"status": "success"}
