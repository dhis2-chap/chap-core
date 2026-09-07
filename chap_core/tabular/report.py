"""Minimal HTML report for a phase-1 evaluation.

Manifest, the list of input assumptions that were checked, the per-fold metric
table, and the headline metric. The report states plainly that the numbers are
cross-validation numbers.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_core.tabular.cv import EvaluationResult

ASSUMPTIONS_CHECKED = [
    "Target column present",
    "No missing values",
    "All columns numeric or encoded",
    "No exact duplicate rows",
    "Target is not constant",
]


def _table(result: EvaluationResult) -> str:
    metrics = list(result.mean)
    header = "".join(f"<th>{html.escape(m)}</th>" for m in metrics)
    body = ""
    for i, fold in enumerate(result.per_fold, start=1):
        cells = "".join(f"<td>{fold[m]:.4f}</td>" for m in metrics)
        body += f"<tr><td>fold {i}</td>{cells}</tr>"
    mean_std = "".join(f"<td>{result.mean[m]:.4f} &plusmn; {result.std[m]:.4f}</td>" for m in metrics)
    body += f"<tr><td><strong>mean &plusmn; std</strong></td>{mean_std}</tr>"
    return f"<table><thead><tr><th></th>{header}</tr></thead><tbody>{body}</tbody></table>"


def _test_section(result: EvaluationResult) -> str:
    if result.test is None:
        return ""
    metrics: dict[str, float] = result.test["metrics"]
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td></tr>" for name, value in metrics.items())
    headline = result.headline_metric
    return f"""
<h2>Held-out test metrics</h2>
<p class="note">These are single train/test split numbers ({result.test["n_train"]} train /
{result.test["n_test"]} test, seed {result.test["seed"]}), not cross-validation. The saved
model was trained on the training split only.</p>
<table><tbody>{rows}</tbody></table>
<p><strong>{html.escape(headline)}</strong> (test): {metrics[headline]:.4f}</p>
"""


def render_report(result: EvaluationResult) -> str:
    """Return the report as an HTML string."""
    headline = result.headline_metric
    cv_scope = "the training split" if result.test is not None else "the whole dataset"
    manifest = {
        "Model": result.model,
        "Task": result.task,
        "Target": result.target,
        "CV samples": result.n_samples,
        "Features": len(result.feature_names),
        "CV": f"{result.n_splits}-fold, seed {result.seed}"
        + (", stratified" if result.task == "classification" else ""),
    }
    manifest_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in manifest.items()
    )
    assumption_items = "".join(f"<li>{html.escape(a)}</li>" for a in ASSUMPTIONS_CHECKED)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Tabular evaluation: {html.escape(result.model)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.note {{ background: #fff3cd; padding: 0.6rem 1rem; border-left: 4px solid #e0a800; }}
</style>
</head>
<body>
<h1>Tabular evaluation</h1>
<p class="note">The cross-validation numbers below are computed on {cv_scope}.</p>
<h2>Manifest</h2>
<table><tbody>{manifest_rows}</tbody></table>
<h2>Input assumptions checked</h2>
<ul>{assumption_items}</ul>
<h2>Per-fold metrics (cross-validation)</h2>
{_table(result)}
<h2>Headline metric</h2>
<p><strong>{html.escape(headline)}</strong>: {result.mean[headline]:.4f} &plusmn; {result.std[headline]:.4f} (cross-validation)</p>
{_test_section(result)}
</body>
</html>
"""


def write_report(result: EvaluationResult, path: str | Path) -> None:
    Path(path).write_text(render_report(result), encoding="utf-8")
