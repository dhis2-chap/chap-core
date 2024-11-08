# Solution 1: Use JSON-file from Prediction App with CHAP Core CLI

If you are following this documentation, we assume that you have downloaded a JSON file using the "Download button" in the Prediction App. If you have not completed this step, you need to [learn how to install the Prediction App and download a JSON file](./prediction-app.md) first

The JSON file you download from the Prediction App should look similar to this:

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

## Get started with CHAP Cli

#### Requirements:
- Docker is installed **AND** running on your computer (Installation instructions can be found at [https://docs.docker.com/get-started/get-docker/](https://docs.docker.com/get-started/get-docker/)).
- Access to the credentials for Google Earth Engine. (Google Service Account Email and Private Key)

## Install CHAP Core CLI

Follow [the installation instructions to install the chap-core package](../installation).
After installation, the chap command line interface (CLI) should be available in your terminal.

## Credentials for Google Earth Engine

You need to have credentials for Google Earth Engine. We recommend you create a new folder named "chap-core-cli" where you will later run CHAP Core. Inside
this folder, create a new file named ".env" with the two environment variables: **"GOOGLE_SERVICE_ACCOUNT_EMAIL"** and **"GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY"**.

<div style="border-radius: 1px; border-style: dotted; border-color: orange; padding: 10px; color : black; background-color: white; margin-top: 20px; margin-bottom: 20px">
**Note**: It's recommended to use a terminal or a code editor when creating the ".env" file. If you, for instance, are using Windows File Explorer to create this file, you may end up with a text file named .env.txt instead.

</div>

The file should look similar to the following content:

```bash
GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-service-account@company.iam.gserviceaccount.com"
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
```

Convert the JSON data into a CHAP-DataSet
-----------------------------------------
After downloading the JSON data from the Prediction App, it's practical to first convert the data into a CHAP-DataSet. This
fetches the climate data from the Google Earth Engine dataset ERA5-Land Daily Aggregated and harmonizes the data into a single dataSet.This is done by running the following command (replace the placeholders with the actual values):

```bash
chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv
```

## Evaluate models on the dataset
The next step is to evaluate existing models on the dataset to see if some of them perform well on your dataset. Ensure Docker is running.
This is done by running the following command:

```bash
chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars_monthly
```

This will generate a report with the evaluation results for the specified model (in this case chap_ewars).

## Predict using the best model
After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the
predict command:

```bash
chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id chap_ewars_monthly --do-summary
```

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file `~/Downloads/chap_request_data_2024-09-24_two_provinces.json`):

```bash
chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars_monthly
chap-cli predict training_data.csv predictions.csv --model-id chap_ewars_monthly --do-summary
```


