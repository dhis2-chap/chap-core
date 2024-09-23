Running on JSON data downloaded from the CHAP-app
=================================================

Convert the JSON data into a CHAP-DataSet
------------------------------------------

After downloading the JSON data from the CHAP-app, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from DHIS2 and harmonizes the data into a single DataSet. This is done by running the following
command:

```bash
chap-cli convert <path-to-json-file> <path-to-output-file>
```

Evaluate models on the dataset
------------------------------
The next step is to evaluate existing models on the dataset, to see if some of them perform well on your dataset.
This is done by running the following command:

```bash
chap-cli evaluate <path-to-dataset-file> <path-to-report-file>.pdf --model-path <model-name>
```

This will generate a report with the evaluation results for the specified model. The default model path is:

Predict using the best model
----------------------------

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

```bash
chap-cli predict <path-to-dataset-file> <path-to-output-file> --model-path <model-name>
```

