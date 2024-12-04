

class ModelFailedException(Exception):
    ...


class InvalidModelException(Exception):
    ...


class CommandLineException(Exception):
    ...


class NoPredictionsError(Exception):
    pass

class GEEError(Exception):
    pass


class ModelConfigurationException(Exception):
    ...


class InvalidDateError(Exception):
    ...