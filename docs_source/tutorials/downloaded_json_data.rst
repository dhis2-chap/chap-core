Running on JSON data downloaded from the CHAP-app
=================================================

Convert the JSON data into a CHAP-DataSet
------------------------------------------

After downloading the JSON data from the CHAP-app, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from DHIS2 and harmonizes the data into a single DataSet. This is done by running the following
command (replace the placeholders with the actual values):

.. code-block:: bash

    chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv


Evaluate models on the dataset
------------------------------
The next step is to evaluate existing models on the dataset, to see if some of them perform well on your dataset.
This is done by running the following command:

.. code-block:: bash

    chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars


This will generate a report with the evaluation results for the specified model (in this case chap_ewars).

Predict using the best model
----------------------------

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

.. code-block:: bash

    chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id <model-name> --do-summary

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file ~/Downloads/chap_request_data_2024-09-24_two_provinces.json):

.. code-block:: bash

    chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
    chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars
    chap-cli predict training_data.csv predictions.csv --model-id chap_ewars --do-summary
