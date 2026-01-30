"""Hatchling build hook to generate CLI constants at build time."""

import re
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class GenerateConstantsHook(BuildHookInterface):
    """Build hook that generates constants for CLI help text."""

    PLUGIN_NAME = "generate-constants"

    def initialize(self, version, build_data):
        root = Path(self.root)
        backtest_plots_dir = root / "chap_core" / "assessment" / "backtest_plots"
        plot_ids = self._extract_plot_ids(backtest_plots_dir)

        output_path = root / "chap_core" / "cli_endpoints" / "_generated.py"
        output_path.write_text(
            f'"""Auto-generated at build time. Do not edit."""\n\n'
            f"PLOT_TYPES = {plot_ids!r}\n"
        )

        build_data["artifacts"].append("chap_core/cli_endpoints/_generated.py")

    def _extract_plot_ids(self, directory: Path) -> list[str]:
        """Extract plot IDs from @backtest_plot decorators using regex."""
        pattern = re.compile(r'@backtest_plot\s*\(\s*id\s*=\s*["\']([^"\']+)["\']')
        plot_ids = []

        for py_file in directory.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text()
            matches = pattern.findall(content)
            plot_ids.extend(matches)

        return sorted(plot_ids)
