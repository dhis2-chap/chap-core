import dataclasses
from asyncio import CancelledError
from typing import Optional

from chap_core.worker.interface import Job


class Control:
    def __init__(self, controls):
        self._controls = controls
        self._status = "idle"
        self._current_control = None
        self._is_cancelled = False

    @property
    def current_control(self):
        return self._current_control

    def cancel(self):
        if self._current_control is not None:
            self._current_control.cancel()
        self._is_cancelled = True

    def set_status(self, status):
        self._current_control = self._controls.get(status, None)
        self._status = status
        if self._is_cancelled:
            raise CancelledError()

    def get_status(self):
        if self._current_control is not None:
            return f"{self._status}:  {self._current_control.get_status()}"
        return self._status

    def get_progress(self):
        if self._current_control is not None:
            return self._current_control.get_progress()
        return 0


@dataclasses.dataclass
class InternalState:
    control: Optional[Control]
    current_data: dict
    model_path: Optional[str] = None
    current_job: Job | None = None

    def is_ready(self):
        return self.current_job is None or self.current_job.is_finished
