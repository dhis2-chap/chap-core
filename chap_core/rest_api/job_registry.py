"""Functions a Celery job may run, and how their arguments cross the JSON broker.

Messages carry a registered function name and JSON arguments, never a pickled function,
so the worker can only run what is registered here. Kept free of Celery imports so the
CLI can import the job modules without loading the worker setup.
"""

import inspect
from collections.abc import Callable
from typing import Any, get_args, get_type_hints

from pydantic import BaseModel, TypeAdapter

JOB_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def _name_of(func: Callable[..., Any]) -> str:
    return f"{func.__module__}.{func.__qualname__}"


def job_function[JobFunction: Callable[..., Any]](func: JobFunction) -> JobFunction:
    """Register a function so CeleryPool can queue it."""
    JOB_FUNCTIONS[_name_of(func)] = func
    return func


def job_name_for(func: Callable[..., Any]) -> str:
    """Name to send for a registered function; raises for one that is not registered."""
    name = _name_of(func)
    if JOB_FUNCTIONS.get(name) is not func:
        raise ValueError(f"{name} is not registered with @job_function and cannot be queued")
    return name


def _to_json_arg(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_to_json_arg(item) for item in value]
    return value


def json_args(args: tuple) -> list[Any]:
    """Turn pydantic models in positional arguments into dicts the JSON broker can carry."""
    return [_to_json_arg(value) for value in args]


def json_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Turn pydantic models in keyword arguments into dicts the JSON broker can carry."""
    return {key: _to_json_arg(value) for key, value in kwargs.items()}


def _contains_model(annotation: Any) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return any(_contains_model(arg) for arg in get_args(annotation))


def resolve_job(func_name: str, args: tuple, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple, dict]:
    """Look up a queued function and rebuild the pydantic arguments its type hints ask for."""
    func = JOB_FUNCTIONS.get(func_name)
    if func is None:
        raise ValueError(f"Unknown job function {func_name!r}")
    bound = inspect.signature(func).bind_partial(*args, **kwargs)
    hints = get_type_hints(func)
    for name, value in bound.arguments.items():
        if name in hints and _contains_model(hints[name]):
            bound.arguments[name] = TypeAdapter(hints[name]).validate_python(value)
    return func, bound.args, bound.kwargs
