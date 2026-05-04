import os
import threading
from typing import Any

_surrogate_cache: dict[tuple, Any] = {}
_surrogate_cache_lock = threading.Lock()
_SURROGATE_CACHE_MAX = int(os.getenv("CHAP_SURROGATE_CACHE_MAX", "20"))


def get_cached_surrogate(key: tuple) -> Any | None:
    with _surrogate_cache_lock:
        return _surrogate_cache.get(key)


def put_cached_surrogate(key: tuple, explainer: Any) -> None:
    with _surrogate_cache_lock:
        while len(_surrogate_cache) >= _SURROGATE_CACHE_MAX:
            oldest = next(iter(_surrogate_cache))
            del _surrogate_cache[oldest]
        _surrogate_cache[key] = explainer
