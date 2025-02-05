# Setting Up CHAP Core as a Contributor

The following is our recommended setup for creating a development environment when working with CHAP Core as a contributor. 

1. If you're a Windows user, start by [reading this note for Windows users](../contributor/windows_contributors) and initiate a wsl session from the commandline:

    ```bash
    $ wsl
    ```

2. Clone the chap-core dev branch to a folder of your choice:

    ```bash
    $ git clone https://github.com/dhis2-chap/chap-core/tree/dev
    $ cd chap-core
    ```

3. Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it. We use uv to manage the development environment. 
The benefit of uv is that it makes installing dependencies faster. 
To read more, check out [their documentation](https://docs.astral.sh/uv/getting-started/installation/).

    * Start by installing uv as per the official documentation:

      ```bash
      $ curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

    * Remember to restart the linux shell (or wsl if you're on windows) for the uv command to become available

4. Install a local version of Python along with all the dependencies. Inside the project folder, run:

      ```bash
      $ uv sync --dev
      ```

    Note that uv creates a virtual Python environment for you, so you don’t need to create one yourself. 
    This environment exists in the `.venv` directory. 

5. Activate the environment and run the tests to make sure everything is working:

      ```bash
      $ source .venv/bin/activate 
      $ pytest
      ```

    We recommend a setup where you can run the tests directly through the IDE you are using (e.g. Vscode or Pycharm). 
    Make sure that your IDE is using the correct Python environment.

6. Finally, if the tests are passing, you should now be connected to the development version of Chap, directly reflecting 
any changes you make to the code. Check to ensure that the chap command line interface (CLI) is available in your terminal:

      ```bash
      $ chap-cli --help
      ```

You have now successfully setup a development version of the chap-cli tool and you are ready to start developing. 
If you have any problems installing or setting up the environment, feel free to [contact us](https://github.com/dhis2-chap/chap-core/wiki>). 