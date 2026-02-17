"""Console script for chap_core."""

import logging
import warnings

# Suppress GluonTS deprecation warning about pandas frequency aliases
warnings.filterwarnings("ignore", category=FutureWarning, module="gluonts")
# Suppress bionumpy warnings
warnings.filterwarnings("ignore", module="bionumpy")

from cyclopts import App

from chap_core.cli_endpoints import convert, evaluate, forecast, preference_learn, utils, validate

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = App()

# Register commands from each module
convert.register_commands(app)
evaluate.register_commands(app)
forecast.register_commands(app)
preference_learn.register_commands(app)
utils.register_commands(app)
validate.register_commands(app)


def main():
    app()


if __name__ == "__main__":
    app()
