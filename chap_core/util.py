from shutil import which
import numpy as np
import os


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x = nan_helper(y)
        >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def conda_available():
    return which("conda") is not None


def docker_available():
    return which("docker") is not None


def pyenv_available():
    return which("pyenv") is not None


def redis_available():
    try:
        r = load_redis()
        r.ping()
        return True
    except Exception as e:
        if e.__class__.__name__ in ("ModuleNotFoundError", "ConnectionError"):
            return False
        else:
            # Handle other exceptions
            raise


def load_redis(db=0):
    import redis

    host = os.getenv("REDIS_HOST", "localhost")  # default to localhost for backward compatibility
    port = os.getenv("REDIS_PORT", "6379")
    r = redis.Redis(host=host, port=int(port), db=db)
    return r
