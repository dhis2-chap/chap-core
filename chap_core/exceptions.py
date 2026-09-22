class ModelFailedException(Exception): ...


class InvalidModelException(Exception): ...


class CommandLineException(Exception): ...


class NoPredictionsError(Exception):
    pass


class GEEError(Exception):
    pass


class ModelConfigurationException(Exception): ...


class InvalidDateError(Exception): ...


class ChapkitServiceStartupError(Exception):
    """Raised when a chapkit model service fails to start."""


class DockerUnavailableError(Exception):
    """Raised when the Docker daemon cannot be reached."""


class ModelTemplateRevisionConflict(ValueError):
    """A model template version label points at a different source revision than the one stored.

    A version is write-once and its stored digest is what earlier backtests ran against, so
    the new revision needs a new version label instead.
    """

    def __init__(
        self, name: str, version: str, stored_digest: str | None, reported_digest: str | None, how_to_fix: str
    ):
        self.name = name
        self.version = version
        self.stored_digest = stored_digest
        self.reported_digest = reported_digest
        super().__init__(
            f"Model template {name!r} version {version!r} is stored from revision {stored_digest!r}, "
            f"but its source now reports revision {reported_digest!r}. A version is write-once, so "
            f"the new revision needs a new version label: {how_to_fix}"
        )
