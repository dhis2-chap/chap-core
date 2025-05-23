# Running on JSON Data Downloaded from the DHIS2 Modeling App

**Use this instruction when CHAP Core is not installed at a backend service on your DHIS2-instance**

## Convert the JSON Data into a CHAP-DataSet

After downloading the JSON data from the Modeling App, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from the Google Earth Engine dataset ERA5-Land Daily Aggregated and harmonizes the data into a single DataSet.
This is done by running the following command (replace the placeholders with the actual values):

```bash
$ chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv
```

## Evaluate Models on the Dataset

The next step is to evaluate existing models on the dataset to see if some of them perform well on your dataset. Ensure Docker is running.
This is done by running the following command:

```bash
$ chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars
```

This will generate a report with the evaluation results for the specified model (in this case `chap_ewars`).

## Predict Using the Best Model

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

```bash
$ chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id chap_ewars --do-summary
```

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file `~/Downloads/chap_request_data_2024-09-24_two_provinces.json`):

```bash
$ chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
$ chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars
$ chap-cli predict training_data.csv predictions.csv --model-id chap_ewars --do-summary
```
