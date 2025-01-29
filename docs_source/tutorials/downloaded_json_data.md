# Running on JSON Data Downloaded from the Prediction App

**Use this instruction when CHAP Core is not installed at a backend service on your DHIS2-instance**

### Requirements
- Docker is installed **and running** on your computer ([Installation instructions](https://docs.docker.com/get-started/get-docker/)).
- Prediction App is installed on your DHIS2-instance ([Instruction for Prediction App installation](https://github.com/dhis2/prediction-app)).
- Access to credentials for Google Earth Engine (Google Service Account Email and Private Key).

## Install CHAP Core

Follow [the installation instructions to install the chap-core package](<installation>). After installation, the CHAP command-line interface (CLI) should be available in your terminal.

We recommend you run CHAP Core with Conda. If you don't have Conda, you can install Miniconda 
([minimal installer for Conda](https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links)).

## Credentials for Google Earth Engine

You need to have credentials for Google Earth Engine. We recommend you create a new folder where you will later run CHAP Core. Inside 
this folder, create a new file named `.env` with the two environment variables: **"GOOGLE_SERVICE_ACCOUNT_EMAIL"** and **"GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY"**. The file should look similar to the following content:

```
GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-serviec-account@company.iam.gserviceaccount.com"
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
```

## Convert the JSON Data into a CHAP-DataSet

After downloading the JSON data from the Prediction App, it's practical to first convert the data into a CHAP-DataSet. This 
fetches the climate data from the Google Earth Engine dataset ERA5-Land Daily Aggregated and harmonizes the data into a single DataSet. 
This is done by running the following command (replace the placeholders with the actual values):

```
chap-cli harmonize <path-to-json-file>.json <path-to-output-file>.csv
```

## Evaluate Models on the Dataset

The next step is to evaluate existing models on the dataset to see if some of them perform well on your dataset. Ensure Docker is running. 
This is done by running the following command:

```
chap-cli evaluate <path-to-dataset-file>.csv <path-to-report-file>.pdf --model-id chap_ewars
```

This will generate a report with the evaluation results for the specified model (in this case `chap_ewars`).

## Predict Using the Best Model

After evaluating the models, you can predict the values for the dataset using the best model. This is done by running the 
predict command:

```
chap-cli predict <path-to-dataset-file>.csv <path-to-output-file>.csv --model-id chap_ewars --do-summary
```

This will generate a new dataset with the predicted values for the dataset.

An example of the full workflow would be (for input file `~/Downloads/chap_request_data_2024-09-24_two_provinces.json`):

```
chap-cli harmonize ~/Downloads/chap_request_data_2024-09-24_two_provinces.json training_data.csv
chap-cli evaluate training_data.csv evaluation_report.pdf --model-id chap_ewars
chap-cli predict training_data.csv predictions.csv --model-id chap_ewars --do-summary
```
