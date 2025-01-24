Setting up CHAP-Core CLI tool
---------------------------

If you are developing custom models or integrating external forecasting models with CHAP, you should install the 
chap-core Python package. 

NOTE: This guide is meant for end-users. If you are a developer and want to contribute to the CHAP Core codebase, you 
should follow the :ref:`getting started guide for developers instead </../developer/getting_started.md>`_. 

We recommend installing chap-core inside a Python `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ 
to avoid conflicts with other Python packages. 
On Windows, the best approach is to use Conda to manage environments. If you don't have Conda, you can install Miniconda,
(a minimal installer for Conda) from https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links

- Windows: After installation open "Anaconda Prompt". Search for "Anaconda Prompt" in the Windows Start menu.
- Linux: Conda should work in your default terminal after installation.

Using Conda, you can create an environment like this:

    $ conda create -n chap-core python=3.11

    $ conda activate chap-core

After activating the environment, intalling chap-core can be done easily using pip:

.. code-block:: console

    $ pip install chap-core

To verify that the installation worked, check that you have the `chap` and `chap-cli` commands available in yor terminal. 
For instance, typing `chap-cli` should give something like:

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
