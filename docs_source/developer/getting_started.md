# Developer getting started

The main intended way of contributing to CHAP-Core is by contributing with models, for which we have a modularized system that makes it easy to contribute.
For this, we have guides/tutorials that [explain how to make models compatible with CHAP](../external_models/making_external_models_compatible.rst).

We are also working on adding similar guides for contributing with custom code for evaluating models and visualizing results.
The code for evaluating and visualizing results is currently tightly integrated into the chap-core code base, but the plan is to 
make this more modularized and easier to contribute to.

This document describes how to get started for contributing to the chap-core code base itself. 


## Getting started working with the chap-core codebase

Contributing to the chap-core code base requires that you have a clone of the chap-core repository.

It can also be useful to see [the code overview guide](code_overview.rst) for a brief overview of the code base.

### Development setup

If you want to contribute to the chap-core Python codebase, you will need to set up a development environment. 
The following is our recommended setup. You will need to have Python 3.10 or a higher version installed.

1. Due to the limited support for Windows in many of the dependencies and to ensure a consistent development environment, 
Windows users should use wsl to operate in a Linux environment. If this is the first time you're using wsl on Windows:

    * First create a wsl linux environment with `wsl install`

    * Make docker available from within the wsl environment:

      * In Docker Desktop, go to Settings - Resources - WSL Integration and check off the Linux distro used by wsl, e.g. `Ubuntu`

    * Enter the linux environment with `wsl`

    * Now you should be ready to follow the remaining steps below

2. Clone the chap-core dev branch to a folder of your choice:

    ```
    $ git clone https://github.com/dhis2-chap/chap-core/tree/dev
    $ cd chap-core
    ```

3. Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it. 
We use uv to manage the development environment. 
The benefit of uv is that it makes installing dependencies faster. 
To read more, check out [their documentation](https://docs.astral.sh/uv/getting-started/installation/).

    * Start by installing uv as per the official documentation:

      ```
      $ curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

    * Remember to restart the linux shell (or wsl if you're on windows) for the uv command to become available

4. Install the dependencies. Inside the project folder, run:

      ```
      $ uv sync --dev
      ```

Note that uv creates a virtual environment for you, so you don’t need to create one yourself. 
This environment exists in the `.venv` directory. 

5. Activate the environment and run the tests to make sure everything is working:

      ```
      $ source .venv/bin/activate 
      $ pytest
      ```

We recommend a setup where you can run the tests directly through the IDE you are using (e.g. Vscode or Pycharm). 
Make sure that your IDE is using the correct Python environment.

6. Finally, if the tests are passing, you should now be connected to the development version of Chap, directly reflecting 
any changes you make to the code. Check to ensure that the chap command line interface (CLI) is available in your terminal:

      ```
      $ chap-cli --help
      ```

At this point you have a development version of the chap-cli tool and you are ready to start developing. 
If you have any problems installing or setting up the environment, feel free to [contact us](https://github.com/dhis2-chap/chap-core/wiki>). 

See more about testing in the [testing guide](testing.rst).


### Contributing code

Code contributions should mainly happen by creating a pull request in the chap-core repository. In order to do this, you
will have to have a clone of the chap-core repository on github (which is possible for anyone with a github account).

Some internal developers can also push directly to the main chap-core repository.

For an overview of the code base, see the [code overview](code_overview) guide.
