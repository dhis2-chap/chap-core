# Setting Up CHAP Core as a Contributor

The following is our recommended setup for creating a development environment when working with CHAP Core as a contributor.

If you are an external contributor without write-access to the chap-core repository you will first need to [fork the chap-core repository to your own GitHub account](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo?tool=webui).

1.  Start by downloading the latest chap-core `dev` branch to a local folder of your choice, either from the main repository or from your own fork:

    ```bash
    $ git clone https://github.com/dhis2-chap/chap-core/tree/dev
    $ cd chap-core
    ```

2.  If you need to work with and test a specific stable version of CHAP Core codebase, these are stored as version tags. Writing `git tag` on the commandline will give you a list of the available version. To switch to a desired version, for instance v1.0.3, you can write:

    ```bash
    git switch tags/v1.0.3
    ```

3.  If you're a Windows user, [read this note about how to simulate a Linux environment using Windows WSL](../contributor/windows_contributors). Before proceeding to the next steps, initiate a wsl session from the commandline:

    ```bash
    $ wsl
    ```

4.  Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it. The benefit of using `uv` for the development environment is that it makes installing dependencies much faster.
    To read more, check out [their documentation](https://docs.astral.sh/uv/getting-started/).

        * Fetch and install `uv` from their official website:

          ```bash
          $ curl -LsSf https://astral.sh/uv/install.sh | sh
          ```

        * After the installation, restart the linux shell (or wsl if you're on windows) in order for the uv command to become available.

5.  Install a local version of Python along with all the dependencies. Inside the project folder, run:

    ```bash
    $ uv sync --dev
    ```

    Note that `uv` creates a virtual Python environment with all the required packages for you, so you don’t need to do this manually.
    This environment exists in the `.venv` directory.

6.  Activate the environment and run the tests to make sure everything is working:

    ```bash
    $ source .venv/bin/activate
    $ pytest
    ```

    We recommend a setup where you can run the tests directly through the IDE you are using (e.g. Vscode or Pycharm).
    Make sure that your IDE is using the correct Python environment.

7.  Finally, if the tests are passing, you should now be connected to the development version of Chap, directly reflecting
    any changes you make to the code. Check to ensure that the chap command line interface (CLI) is available in your terminal:

          ```bash
          $ chap --help
          ```

It is also a good to see if you can run chap evaluation on an external model, by running:

```bash
chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil
```

If the above command runs successfully, a report.pdf file will be generated with the results.

You have now successfully setup a development version of the chap-cli tool and you are ready to start developing.
If you have any problems installing or setting up the environment, feel free to [contact us](https://github.com/dhis2-chap/chap-core/wiki>).
