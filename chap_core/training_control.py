from asyncio import CancelledError


class TrainingControl:
    def __init__(self):
        self._total_samples = None
        self._cancelled = False
        self._status = "None"
        self._n_finished = 0

    def set_total_samples(self, total_samples):
        self._total_samples = total_samples

    def get_progress(self):
        return self._n_finished / self._total_samples if self._total_samples is not None else 0

    def get_status(self):
        return self._status

    def register_progress(self, n_sampled):
        if self._cancelled:
            raise CancelledError()
        self._n_finished += n_sampled

    def set_status(self, status):
        if self._cancelled:
            raise CancelledError()
        self._status = status

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


class PrintingTrainingControl(TrainingControl):
    def register_progress(self, n_sampled):
        super().register_progress(n_sampled)
        print(f"Progress: {self.get_progress() * 100:.2f}%")

    def set_status(self, status):
        super().set_status(status)
        print(f"Status: {self.get_status()}")
