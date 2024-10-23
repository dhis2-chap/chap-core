Running on JSON data downloaded from the CHAP-app
=============================================================================================================

**Use this instruction when CHAP Core is not installed at a backend service on your DHIS2-instance**

Requriments: 
    - Docker is installed AND running on your computer (Installation instructions can be found at https://docs.docker.com/get-started/get-docker/).
    - CHAP-app is installed on your DHIS2-instance (Instruction for CHAP-app installation could be found at https://github.com/dhis2/chap-app)
    - Access to credentials for Google Earth Engine. (Google Service Account Email and Private Key)

Install CHAP Core
-----------------
We recommend you run CHAP Core with Conda. If you don't have Conda, you could install Miniconda, 
(a minimal installer for Conda) from https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links

- Windows: After installation open "Anaconda Prompt". Search for "Anaconda Prompt" in the Windows Start menu.
- Linux: Conda should work in your default terminal after installation.

**We recommend you to create a new conda environment by running the following commands:**

    $ conda create -n chap-core python=3.11

    $ conda activate chap-core

**In the same shell, install CHAP Core, by runing the following command (10-20 min):**

    $ pip install git+https://github.com/dhis2/chap-core.git

After installation, chap command line interface (CLI) should be available in your terminal.

Credentials for Google Earth Engine
------------------------------------------
You need to have credentials for Google Earth Engine. We recommend you to create a new folder where you will later run CHAP Core. Inside 
this folder, create a new file named ".env" with the two environment variables: **"GOOGLE_SERVICE_ACCOUNT_EMAIL"** and **"GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY"**. 
The file should look similar to the following content:

.. code-block:: bash

    GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-serviec-account@company.iam.gserviceaccount.com"
    GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"

Convert the JSON data into a CHAP-DataSet
------------------------------------------

After downloading the JSON data from the CHAP-app, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from the Google Earth Engine dataset ERA5-Land Daily Aggregated and harmonizes the data into a single DataSet. 
This is done by running the following command (replace the placeholders with the actual values):

.. code-block:: bash

    chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv


Evaluate models on the dataset
------------------------------
The next step is to evaluate existing models on the dataset, to see if some of them perform well on your dataset. Ensure Docker is running.
This is done by running the following command:

.. code-block:: bash

    chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars


This will generate a report with the evaluation results for the specified model (in this case chap_ewars).

Predict using the best model
----------------------------

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

.. code-block:: bash

    chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id chap_ewars --do-summary

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file ~/Downloads/chap_request_data_2024-09-24_two_provinces.json):

.. code-block:: bash

    chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
    chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars
    chap-cli predict training_data.csv predictions.csv --model-id chap_ewars --do-summary
