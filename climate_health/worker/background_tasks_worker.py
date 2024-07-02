from typing import Generic

from fastapi import BackgroundTasks

from .interface import ReturnType


class BGTaskJob(Generic[ReturnType]):
    def __init__(self, state, job_id):
        self._state = state
        self._job_id = job_id

    @property
    def status(self):
        return self._state.control.get_status()

    @property
    def progress(self):
        return self._state.control.get_progress()

    @property
    def result(self):
        return self._result_dict

    def cancel(self):
        self._state.control.cancel()

    @property
    def is_finished(self):
        return self._job_id in self._result_dict


class BGTaskWorker(Generic[ReturnType]):
    def __init__(self, background_tasks: BackgroundTasks, internal_state: 'InternalState'):
        self._background_tasks = background_tasks
        self._result_dict = internal_state.current_data
        self._state = internal_state

    def new_id(self):
        self._id += 1
        return self._id

    def wrapper(self, func, job_id):
        def wrapped(*args, **kwargs):
            self._result_dict[job_id] = func(*args, **kwargs, control=self._state.control)

        return wrapped

    def queue(self, func, *args, **kwargs) -> BGTaskJob[ReturnType]:
        job_id = self.new_id()
        self._background_tasks.add_task(self.wrapper(func, job_id), *args, **kwargs)
        return BGTaskJob(self._state, job_id)
