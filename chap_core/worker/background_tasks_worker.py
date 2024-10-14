from asyncio import CancelledError
from typing import Generic

from fastapi import BackgroundTasks

from .interface import ReturnType
from ..internal_state import InternalState, Control
from ..training_control import TrainingControl


class BGTaskJob(Generic[ReturnType]):
    def __init__(self, state, job_id):
        self._state = state
        self._job_id = job_id
        self._result_dict = state.current_data

    @property
    def status(self):
        return self._state.control.get_status()

    @property
    def progress(self):
        return self._state.control.get_progress()

    @property
    def result(self):
        return self._result_dict[self._job_id]

    def cancel(self):
        self._state.control.cancel()

    @property
    def is_finished(self):
        return self._job_id in self._result_dict


class BGTaskWorker(Generic[ReturnType]):
    def __init__(self, background_tasks: BackgroundTasks, internal_state: InternalState, state):
        self._background_tasks = background_tasks
        self._result_dict = internal_state.current_data
        self._state = internal_state
        self._ready_state = state
        self._id = 0

    def new_id(self):
        self._id += 1
        return self._id

    def wrapper(self, func, job_id):
        def wrapped(*args, **kwargs):
            self._state.control = Control({"Training": TrainingControl()})
            try:
                print("Started")
                self._result_dict[job_id] = func(*args, **kwargs, control=self._state.control)
                print("Finished")
            except CancelledError:
                self._result_dict[job_id] = None
                self._ready_state.status = "cancelled"
                self._ready_state.ready = True
                self._state.control = None

        return wrapped

    def queue(self, background_tasks, func, *args, **kwargs) -> BGTaskJob[ReturnType]:
        print(f"Queueing task {func.__name__}(args={args}, kwargs={kwargs})")
        job_id = self.new_id()
        print("Job id:", job_id)
        self._background_tasks.add_task(self.wrapper(func, job_id), *args, **kwargs)
        return BGTaskJob(self._state, job_id)
