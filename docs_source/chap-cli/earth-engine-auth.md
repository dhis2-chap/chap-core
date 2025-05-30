# CHAP Core Credentials for Google Earth Engine

In many cases CHAP Core needs credentials for Google Earth Engine in order to connect with the climate data needed for the models to run. Steps to add these credentials:

1. Create or retrieve credentials to Google Earth Engine (Google Service Account Email and Private Key). Read more here: [https://docs.dhis2.org/en/manage/reference/google-service-account-configuration.html](https://docs.dhis2.org/en/manage/reference/google-service-account-configuration.html)

2. Locate the folder containing CHAP Core. This depends if you installed CHAP Core as a [commandline tool](chap-core-setup), as a [local Docker container](docker-compose-doc), or [on a server](running-chap-on-server).

3. Inside this folder, create a new file named `.env` with the following content:

```bash
    GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-service-account@company.iam.gserviceaccount.com"
    GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
```

The file should be stored in the root folder of CHAP Core, such as below:

    /chap-core
    |-- compose.integration.test.yml
    |-- compose.test.yml
    |-- compose.yml
    |-- .env
    |-- chap_core/

_Not all files are included in this file-structure snippet_

<div style="border-radius: 1px; border-style: dotted; border-color: black; background-color: orange; padding: 10px; color: black; background-color: white; margin-top: 20px; margin-bottom: 20px">
**Note**: It's recommended to use a terminal or a code editor when creating the ".env" file. If you, for instance, are using Windows File Explorer to create this file, you may end up with a text file named .env.txt instead.
</div>
