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


class ChapkitServicePortInUseError(ChapkitServiceStartupError):
    """Raised when a chapkit model service exits at startup because its port is taken."""


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
        write_once = "A version is write-once, so the new revision needs a new version label:"
        if reported_digest is None:
            message = (
                f"Model template {name!r} version {version!r}: its source reports no git revision, "
                f"so chap-core cannot check which source it runs. To fix it, {how_to_fix}"
            )
        elif stored_digest is None:
            message = (
                f"Model template {name!r} version {version!r} was stored before chap-core recorded "
                f"source revisions, so the revision {reported_digest!r} its source reports now cannot "
                f"be matched to it. {write_once} {how_to_fix}"
            )
        else:
            message = (
                f"Model template {name!r} version {version!r} is stored from revision {stored_digest!r}, "
                f"but its source now reports revision {reported_digest!r}. {write_once} {how_to_fix}"
            )
        super().__init__(message)


class BaselineTemplateConflict(ValueError):
    """A registered service uses the name of a built-in baseline model template.

    Baselines are built into chap-core, so a service cannot store, update or supersede them.
    """

    def __init__(self, name: str):
        self.name = name
        super().__init__(
            f"Model template {name!r} is a built-in baseline and cannot be stored from a service. "
            "Register the service under another model id."
        )
