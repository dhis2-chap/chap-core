"""Run one chap CLI command for the UI and record its exit code.

Usage: python -m chap_core.ui.job_runner <run_dir> <chap arguments...>
"""

import os
import sys
import traceback
from pathlib import Path


def run(run_dir: Path, args: list[str]) -> int:
    # The UI starts this process without fork-only options (see services.start_job), so do here what
    # they would have done: lead a new session, so stopping the job reaches everything it started,
    # and work inside the run directory, so relative output paths land there.
    if hasattr(os, "setsid"):
        os.setsid()
    os.chdir(run_dir)
    try:
        from chap_core.cli import app

        app(args)
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    (run_dir / "exit_code").write_text(str(code))
    return code


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), sys.argv[2:]))
