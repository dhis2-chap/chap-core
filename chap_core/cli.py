"""Console script for chap_core."""

import logging
import signal
import sys
import warnings

# Suppress GluonTS deprecation warning about pandas frequency aliases
warnings.filterwarnings("ignore", category=FutureWarning, module="gluonts")
# Suppress bionumpy warnings
warnings.filterwarnings("ignore", module="bionumpy")

from cyclopts import App

from chap_core.cli_endpoints import (
    aggregate_eval,
    causal,
    convert,
    ensemble,
    evaluate,
    explain,
    generate_modelcard,
    init,
    model,
    preference_learn,
    report,
    simulate,
    utils,
    validate,
)
from chap_core.exceptions import (
    ChapkitServiceStartupError,
    DockerUnavailableError,
    InvalidModelException,
    ModelConfigurationException,
)
from chap_core.log_config import is_debug_mode
from chap_core.project import project_defaults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s [%(name)s]",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Inside a Chap project (see `chap init`), options not given on the command line come from the project.
app = App(config=[project_defaults])

# Register commands from each module
aggregate_eval.register_commands(app)
causal.register_commands(app)
convert.register_commands(app)
evaluate.register_commands(app)
explain.register_commands(app)
init.register_commands(app)
model.register_commands(app)
preference_learn.register_commands(app)
report.register_commands(app)
simulate.register_commands(app)
utils.register_commands(app)
validate.register_commands(app)
generate_modelcard.register_commands(app)
ensemble.register_commands(app)


# Errors caused by the model or the local setup rather than by a bug in Chap.
USER_FACING_ERRORS = (
    ChapkitServiceStartupError,
    DockerUnavailableError,
    InvalidModelException,
    ModelConfigurationException,
)


def exit_on_signal(signum, frame):
    """Exit like Ctrl-C does, so context managers run and stop model services and containers."""
    raise SystemExit(128 + signum)


def main():
    # A closed terminal (SIGHUP) or `kill` (SIGTERM) would otherwise end the process without cleanup,
    # leaving chapkit services and docker:// containers started for the command running.
    # A signal ignored on start stays ignored, so `nohup chap eval &` keeps running after logout.
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None and signal.getsignal(signum) is not signal.SIG_IGN:
            signal.signal(signum, exit_on_signal)
    try:
        app()
    except USER_FACING_ERRORS as error:
        # The message says what to fix, and a traceback would bury it. CHAP_DEBUG=1 shows the traceback.
        if is_debug_mode():
            raise
        print(f"Error: {error}\n\nSet CHAP_DEBUG=1 to see the full traceback.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
