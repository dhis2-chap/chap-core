# Climate health project repo
CHAP is a platform for forecasting and for assessing forecasts of climate-sensitive health outcomes.
In the early phase, the focus is on vector-borne diseases like malaria and dengue
The platforms is to perform data parsing, data integration, forecasting based on any of multiple supported models, automatic brokering of compatible models for a given prediction context and robust forecast assessment and method comparison. 

The current version has basic data handling functionality in place, and is almost at a stage where it supports running a first external model (EWARS-Plus)

User documentation:

- #Ideally install mamba to make conda environments faster
- #Create environment, the hydromet_dengue is probably okay
- mamba env create -f external_models/hydromet_dengue/env.yml --name hydromet
- #activate the new environment
- conda activate hydromet 
- 

Developer documentation:
- [How to add an external model](external_models/Readme.md)


## Run CHAP with Docker
Ensure you have installed Docker on your local machine. If not, follow the installation guide [here](https://www.docker.com/get-started/).

### Option 1 - Docker compose (easiest way)

#### Step 1 - select which model to run:
A file named ".env" is located at root-level in this project. This file contains the file path to the R-model that would be run when the CHAP starts. The default value is "./external_models/ewars_Plus", but you could edit this as you want.

#### Step 2 - start CHAP
On root-level run:
```
docker compose up
```
This would start CHAP and a web interface for interaction with CHAP. Go to http://localhost:4000/ in your browser and a webpage should appear.

To stop the containers, run:
```
docker compose down
```

### Option 2 - run singel container