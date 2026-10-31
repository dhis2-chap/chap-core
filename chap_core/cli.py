"""Console script for chap_core."""

import logging
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


def main():
    app()


if __name__ == "__main__":
    app()
