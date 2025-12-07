"""Console script for chap_core."""

import logging

from cyclopts import App

from chap_core.cli_endpoints import evaluate, forecast, preference_learn, utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = App()

# Register commands from each module
evaluate.register_commands(app)
forecast.register_commands(app)
preference_learn.register_commands(app)
utils.register_commands(app)


def main():
    app()


if __name__ == "__main__":
    app()
