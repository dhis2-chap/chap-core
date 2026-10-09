"""Run another program for the UI without passing on the UI's own file descriptors.

Usage: python -m chap_core.ui.spawn <program> [arguments...]

The UI starts this helper with posix_spawn (see services.run_external), which on macOS cannot
also close inherited descriptors, so the helper closes them and then becomes the program.
"""

import os
import sys

from chap_core.ui.job_runner import close_inherited_descriptors

if __name__ == "__main__":
    close_inherited_descriptors()
    os.execvp(sys.argv[1], sys.argv[1:])
