# Use JSON-file from DHIS2 Modeling App with CHAP Core CLI

This document describes how to use the CHAP Core CLI to analyze the JSON file you get when pressing the "Download button" in the Modeling App. If you have not completed this step, you need to [set up the Modeling App to download a JSON file](../modeling-app/running-chap-on-server.md) first.

The JSON file you download from the Modeling App should look similar to this:

```json
{
  {
    "model_id": "EWARS-month",
    "features": [
      {
        "featureId": "population",
        "dhis2Id": "gJkrZH2vV8k",
        "data": [
          //removed for readability
        ]
      },
      {
        "featureId": "disease",
        "dhis2Id": "GPEAQwddr16",
        "data": [
          //removed for readability
        ]
      }
    ],
    "orgUnitsGeoJson": {
      "type": "FeatureCollection",
      "features": [
        //removed for readability
      ]
    }
  }
}
```

## Convert the JSON data into a CHAP-DataSet

After downloading the JSON data from the Modeling App, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from the Google Earth Engine dataset ERA5-Land Daily Aggregated and harmonizes the data into a single dataSet.This is done by running the following command (replace the placeholders with the actual values):

```bash
$ chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv
```

## Evaluate models on the dataset

The next step is to evaluate existing models on the dataset to see if some of them perform well on your dataset. Ensure Docker is running.
This is done by running the following command:

```bash
$ chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars_monthly
```

This will generate a report with the evaluation results for the specified model (in this case chap_ewars).

## Predict using the best model

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

```bash
$ chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id chap_ewars_monthly --do-summary
```

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file `~/Downloads/chap_request_data_2024-09-24_two_provinces.json`):

```bash
$ chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
$ chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars_monthly
$ chap-cli predict training_data.csv predictions.csv --model-id chap_ewars_monthly --do-summary
```
