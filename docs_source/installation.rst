.. highlight:: shell

.. _installation:

Installation and getting started
===================================

Installation of chap-core
---------------------------

Installation of CHAP depends on how you want to use CHAP. If you are developing custom models or integrating external forecasting models with CHAP, you should install the chap-core Python package:

.. code-block:: console

    $ pip install chap-core

Local or server setup with docker
----------------------------------

If you want to run models that are already available with CHAP locally on a computer or on a server, we recommend setting up CHAP using docker. :doc:`See this guide for setting up CHAP using Docker Compose <docker-compose-doc>`.

Development setup
------------------

If you want to contribute to the chap-core Python codebase, you will need to set up a development environment. The following is our recommended setup. You will need to have Python 3.10 or a higher version installed.

1. Clone the chap-core repository:

.. code-block:: console

    $ git clone git@github.com:dhis2-chap/chap-core.git

2. Make sure you have :ref:`uv installed <https://docs.astral.sh/uv/getting-started/installation/>`_.

3. Install the dependencies. Inside the project folder, run:

.. code-block:: console

    $ uv sync --dev

Note that uv creates a virtual environment for you, so you don't need to create one yourself. This environment exists in the .venv directory. In order to run things through the virtual environment, you can use the `uv run` command. You can also activate the virtual environment manually with `source .venv/bin/activate`.

4. Run the tests to make sure everything is working:

.. code-block:: console

    $ uv run pytest

If the tests are passing, you are ready to start developing. Feel free to check out open issues in the chap-core Github repository. If you have any problems installing or setting up the environment, feel free to :ref:`contact us <https://github.com/dhis2-chap/chap-core/wiki>`_.
