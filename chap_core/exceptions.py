class ModelFailedException(Exception): ...


class InvalidModelException(Exception): ...


class CommandLineException(Exception): ...


class NoPredictionsError(Exception):
    pass


class GEEError(Exception):
    pass


class ModelConfigurationException(Exception): ...


class ConfiguredModelConflictError(Exception):
    """Raised when creating a configured model that already exists (same name)."""


class InvalidDateError(Exception): ...


class ChapkitServiceStartupError(Exception):
    """Raised when a chapkit model service fails to start."""


class DockerUnavailableError(Exception):
    """Raised when the Docker daemon cannot be reached."""
