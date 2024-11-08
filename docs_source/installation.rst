.. highlight:: shell

.. _installation:

Installation and getting started
===================================

Requirements
------------
The chap-core Python package requires Python 3.10 or higher, and is tested to work on Windows, Linux and MacOS. Most of the built-in forecasting models in CHAP require Docker to run, and we recommend also using Docker to setup chap-core for most use-cases (see below). The current builtin-models are not resource-intensive and one should be fine with 4 GB of RAM.

Installation of chap-core
---------------------------

Installation of CHAP depends on how you want to use CHAP. If you are developing custom models or integrating external forecasting models with CHAP, you should install the chap-core Python package, which can be done easily using pip:

.. code-block:: console

    $ pip install chap-core

We recommend installing chap-core inside a Python `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ to avoid conflicts with other Python packages. On Windows, the best approach is to use Conda to manage environments. If you don't have Conda, you can install Miniconda,
(a minimal installer for Conda) from https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links

- Windows: After installation open "Anaconda Prompt". Search for "Anaconda Prompt" in the Windows Start menu.
- Linux: Conda should work in your default terminal after installation.

Using Conda, you can create an environment like this before running `pip install chap-core`:

    $ conda create -n chap-core python=3.11

    $ conda activate chap-core

To verify that the installation worked, check that you have the `chap` and `chap-cli` commands available in yor terminal. For instance, typing chap-cli should give something like:

.. code-block:: console

    Usage: chap-cli COMMAND

    ╭─ Commands ───────────────────────────────────────────────────────────────────────╮
    │ evaluate   Evaluate how well a model would predict on the last year of the given │
    │            dataset. Writes a report to the output file.                          │
    │ harmonize  Harmonize health and population data from dhis2 with climate data     │
    │            from google earth engine.                                             │
    │ predict                                                                          │
    │ --help,-h  Display this message and exit.                                        │
    │ --version  Display application version.                                          │
    ╰──────────────────────────────────────────────────────────────────────────────────╯


Local or server setup with docker
----------------------------------

If you want to run models that are already available with CHAP locally on a computer or on a server, we recommend setting up CHAP using docker. :doc:`See this guide for setting up CHAP using Docker Compose <docker-compose-doc>`.


Setting up CHAP inside an LXD container
----------------------------------------

It is also possible to run a full installation of CHAP inside an LXD container. For an example setup, see the `CHAP LXD container setup <https://github.com/dhis2-chap/infrastructure>`_.

Development setup
------------------

If you want to contribute to the chap-core Python codebase, you will need to set up a development environment. The following is our recommended setup. You will need to have Python 3.10 or a higher version installed.

1. Clone the chap-core repository:

.. code-block:: console

    $ git clone git@github.com:dhis2-chap/chap-core.git

2. Make sure you have `uv installed <https://docs.astral.sh/uv/getting-started/installation/>`_.

3. Install the dependencies. Inside the project folder, run:

.. code-block:: console

    $ uv sync --dev

Note that uv creates a virtual environment for you, so you don't need to create one yourself. This environment exists in the .venv directory. In order to run things through the virtual environment, you can use the `uv run` command. You can also activate the virtual environment manually with `source .venv/bin/activate`.

4. Run the tests to make sure everything is working:

.. code-block:: console

    $ uv run pytest

If the tests are passing, you are ready to start developing. Feel free to check out open issues in the chap-core Github repository. If you have any problems installing or setting up the environment, feel free to `contact us <https://github.com/dhis2-chap/chap-core/wiki>`_.
