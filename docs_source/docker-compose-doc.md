# Docker Compose (CHAP Core)

Starting CHAP Core using Docker Compose is specifically for those who want to use the CHAP Core REST-API, either together with other services or with the Prediction App installed on a DHIS2 server. See documentation for [Prediction App](prediction-app/prediction-app.md) for instructions on how to install the Prediction App.

**Requirements**
- Access to credentials for Google Earth Engine. (Google Service Account Email and Private Key)

## 1. Install Docker and Docker Compose (if not installed)

### 1.1. Install Docker and Docker Compose
Docker is a platform for developing, shipping, and running applications inside containers.

To download and install Docker, visit the official Docker website: [https://docs.docker.com/get-started/get-docker](https://docs.docker.com/get-started/get-docker)

Docker Compose is a tool for defining and running multi-container Docker applications and is required for starting CHAP Core with Docker. To install Docker Compose, follow the instructions provided in the official Docker documentation:

[Install Docker Compose](https://docs.docker.com/compose/install/)


## 2. Clone CHAP Core GitHub-Repository

You need to clone the CHAP Core repository from GitHub. Open your terminal and run the following command:

```sh
git clone https://github.com/dhis2-chap/chap-core.git
```

## 3. Add Credentials for Google Earth Engine

1. Open your terminal and navigate to the "chap-core" repository you cloned:
   
    ```sh
    cd chap-core
    ```

2. Open the "chap-core" repository in your code editor. For example, if you are using Visual Studio Code, you can use the following command in the terminal:
   
    ```sh
    code .
    ```

3. In your code editor, create a new file at the root level of the repository and name it `.env`.

4. Add the following environment variables to the `.env` file. Replace the placeholder values with your actual Google Service Account credentials:
   
    ```bash
    GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-service-account@company.iam.gserviceaccount.com"
    GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
    ```

## 4. Start CHAP Core

At the root level of the repository (the same level you placed the .env-file), run:

```sh
docker-compose up
```

**This will build three images and start the following containers:**

- A REST-API (FastAPI)
- A Redis server
- A worker service

You can go to [http://localhost:8000/docs](http://localhost:8000/docs) to verify that the REST-API is working. A Swagger page, as shown below, should display:

![Swagger UI](_static/swagger-fastapi.png)

## 5. Stop CHAP Core
```sh
docker-compose down
```
