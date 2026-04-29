"""Lock-in test ensuring generated_plot_ids.py stays in sync with @backtest_plot sources.

The runtime registry can be polluted by docs that contain working
@backtest_plot examples (e.g. creating_custom_backtest_plots.md), so the
check is performed against a fresh AST scan of the source tree — the
same scanner the regenerate script uses.
"""

from scripts.regenerate_plot_help import TARGET_FILE, render_module, scan_plot_ids


def test_generated_plot_ids_file_is_current():
    """If this fails, run `make regen-plot-help` and commit the diff."""
    expected = render_module(scan_plot_ids())
    actual = TARGET_FILE.read_text(encoding="utf-8")
    assert actual == expected, (
        "chap_core/cli_endpoints/generated_plot_ids.py is out of sync with "
        "@backtest_plot registrations. Run `make regen-plot-help` and commit the diff."
    )
