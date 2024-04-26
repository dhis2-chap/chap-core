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
