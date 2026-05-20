"""Safe arithmetic expression parsing, validation, and evaluation for counterfactual transforms."""

from __future__ import annotations

import ast

import pandas as pd

_ALLOWED_FUNCS = frozenset({"abs", "round"})

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Call,
    ast.Load,
    # arithmetic operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def validate_expression(expr: str) -> None:
    """Raise ValueError if expr is not a safe arithmetic expression in x."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression '{expr}': {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Disallowed construct in expression '{expr}'")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS and node.id != "x":
            raise ValueError(f"Disallowed name '{node.id}' in expression '{expr}'")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant in expression '{expr}'")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
            raise ValueError(f"Disallowed function call in expression '{expr}'")


def parse_transformations(transformations: list[str]) -> list[tuple[str, str]]:
    """Parse ['col=expr', ...] into [('col', 'expr'), ...].

    Raises ValueError if any entry lacks an '=' separator.
    """
    result = []
    for t in transformations:
        if "=" not in t:
            raise ValueError(f"Transformation '{t}' is not in 'column=expression' format")
        col, expr = t.split("=", 1)
        result.append((col, expr))
    return result


def apply_transformation(series: pd.Series, expr: str) -> pd.Series:
    """Apply expr to each non-NaN value; leave NaN values unchanged."""
    namespace = {"__builtins__": {}, "abs": abs, "round": round}
    result = series.copy()
    mask = ~series.isna()
    result[mask] = series[mask].apply(lambda x: eval(expr, namespace, {"x": x}))
    return result
