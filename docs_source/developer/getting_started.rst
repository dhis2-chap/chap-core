Developer getting started
=========================

The main intended way of contributing to CHAP is by contributing with models, for which we have a modularized system that makes it easy to contribute.
For this, we have guides/tutorials (see :ref:`external_models_overview`) that explain how to make models compatible with CHAP.

We are also working on adding similar guides for contributing with custom code for evaluating models and visualizing results.
The code for evaluating and visualizing results is currently tightly integrated into the chap-core code base, but the plan is to 
make this more modularized and easier to contribute to.

If you want to contribute to the chap-core code base itself, see the following guides:


Getting started working with the chap-core codebase
---------------------------------------------------

Contributing to the chap-core code base requires that you have a clone of the chap-core repository.

See the following guide for setting up the code base locally for development, 
and see the code overview guide for a brief overview of the code base.

Development setup
------------------

If you want to contribute to the chap-core Python codebase, you will need to set up a development environment. 
The following is our recommended setup. You will need to have Python 3.10 or a higher version installed.

1. Clone the chap-core repository:

.. code-block:: console

    $ git clone git@github.com:dhis2-chap/chap-core.git

2. Install a development version of chap-core. 
We use `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ to manage the development environment, but it is also possible to just use `pip` if you prefer that.
The benefit if uv is that it makes installing dependencies faster. If you are interested in getting started with uv, check out `their documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

3. Install the dependencies. Inside the project folder, run:

If using uv:

.. code-block:: console

    $ uv sync --dev

Note that uv creates a virtual environment for you, so you don't need to create one yourself. This environment exists in the .venv directory. 
In order to run things through the virtual environment, you can use the `uv run` command. 
You can also activate the virtual environment manually with `source .venv/bin/activate` and then run commands as usual.

If using  pip, you probably want to first create a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_, activate it and install chap inside the virtual environment.

.. code-block:: console

    $ pip install -e .


4. Run the tests to make sure everything is working:

If using uv:

.. code-block:: console

    $ uv run pytest

or activate the environment that uv uses and then simply run pytest. This is also the way to run the tests if you are using pip:

.. code-block:: console

    $ source .venv/bin/activate
    $ pytest

We recommend a setup where you can run the tests directly through the IDE you are using (e.g. Vscode or Pycharm). Make sure that your IDE is using the correct
Python environment.

If the tests are passing, you are ready to start developing. If you have any problems installing or setting up the environment, feel free to `contact us <https://github.com/dhis2-chap/chap-core/wiki>`_.

See more about testing in the :ref:`testing` guide.


Contributing code
------------------

Code contributions should mainly happen by creating a pull request in the chap-core repository. In order to do this, you
will have to have a clone of the chap-core repository on github (which is possible for anyone with a github account).

Some internal developers can also push directly to the main chap-core repository.

