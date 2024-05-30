class TrainingControl:
    def __init__(self):
        self._total_samples = None
        self._cancelled = False
        self._n_finished = 0

    def set_total_samples(self, total_samples):
        self._total_samples = total_samples

    def get_progress(self):
        return self._n_finished / self._total_samples

    def register_progress(self, n_sampled):
        self._n_finished += n_sampled

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled
