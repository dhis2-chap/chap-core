"""Run one chap CLI command for the UI and record its exit code.

Usage: python -m chap_core.ui.job_runner <run_dir> <chap arguments...>
"""

import contextlib
import os
import sys
import traceback
from pathlib import Path


def close_inherited_descriptors() -> None:
    """Close every file descriptor inherited from the UI except stdin, stdout and stderr.

    The UI starts its helpers with posix_spawn and close_fds=False (see services.start_job), so
    inheritable descriptors such as the server's listening socket come along. Holding on to that
    socket would keep the UI's port busy until the job ends, so it is closed straight away.
    """
    for fd_dir in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(fd_dir):
            descriptors = [int(name) for name in os.listdir(fd_dir) if name.isdigit()]
            break
    else:
        descriptors = list(range(3, 1024))
    for fd in descriptors:
        if fd > 2:
            with contextlib.suppress(OSError):
                os.close(fd)


LOCK_NAME = "running.lock"
_running_lock = None


def hold_running_lock(run_dir: Path) -> None:
    """Lock running.lock in the run directory for as long as this process lives.

    The system releases the lock when the process ends, however it ends, so the UI can tell a
    running job from a dead one even after a restart, without trusting a process id that may
    have been reused (see services.load_job).
    """
    global _running_lock
    try:
        import fcntl
    except ImportError:  # Windows: the UI falls back to the process id
        return
    _running_lock = open(run_dir / LOCK_NAME, "w")
    fcntl.flock(_running_lock, fcntl.LOCK_EX)


def run(run_dir: Path, args: list[str]) -> int:
    close_inherited_descriptors()
    # The UI starts this process without fork-only options (see services.start_job), so do here what
    # they would have done: lead a new session, so stopping the job reaches everything it started,
    # and work inside the run directory, so relative output paths land there.
    if hasattr(os, "setsid"):
        os.setsid()
    os.chdir(run_dir)
    hold_running_lock(run_dir)
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
