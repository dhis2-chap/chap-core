"""Experimental-marker utilities for REST endpoints."""

from collections.abc import Callable

EXPERIMENTAL_NOTE = "⚠️ **Experimental:** behavior and response shape may change without notice."


def api_experimental[F: Callable](func: F) -> F:
    """Prepend the experimental notice to the route function's docstring.

    Must be applied *below* the route decorator so FastAPI reads the
    updated docstring when registering the route.

    Note: if the route decorator passes an explicit ``description=`` kwarg,
    that kwarg **always wins** and this decorator becomes a silent no-op —
    FastAPI only falls back to the docstring when ``description`` is None.
    Pick one: either this decorator or ``description=``, never both.
    """
    existing = (func.__doc__ or "").strip()
    func.__doc__ = f"{EXPERIMENTAL_NOTE}\n\n{existing}" if existing else EXPERIMENTAL_NOTE
    return func
