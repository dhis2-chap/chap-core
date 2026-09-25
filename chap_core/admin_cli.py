"""Console script for chap-admin: operate a running CHAP instance over its REST API.

Kept apart from ``chap`` because these commands need a running server, while the ``chap``
commands run in-process. Imports stay light so the command starts fast.
"""

import logging

from cyclopts import App

from chap_core.cli_endpoints import marketplace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s [%(name)s]",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = App(name="chap-admin", help="Administer a running CHAP instance: install marketplace models into it.")
app.command()(marketplace.install)
app.command()(marketplace.update)
app.command()(marketplace.uninstall)


def main():
    app()


if __name__ == "__main__":
    app()
