# Read me

## Probabilistic seasonal dengue forecasting in Vietnam using superensembles

### R code
The `R` code presented here provides all the scripts required to run the superensemble of probabilistic dengue models 
presented in the paper _"Probabilistic seasonal dengue forecasting in Vietnam: A modelling study using superensembles"_ 
published in _PLoS Medicine_ by FJ Colón-González et al. (2021). The code comprises seven ordered sub-routines. Each sub-routine has 
a number that indicates its location in a sequence of steps starting at _00_ and ending at _06_. The sub-routine called 
**06_Run_routines.R** contains all the commands required to load and perform all operations in sequence. We suggest 
users only run that script. A description of each sub-routine is provided below.

**- 00_Functions.R:** Loads all developer-defined functions required for the correct functionality of the system.

**- 01_Load_packages.R:** Installs (if not already installed) and loads all the packages required for the correct
functionality of the system.

**- 02_Monthly_data.R:** Reads annual observations of land-cover and population data, and generates monthly time 
series for their incorporation into the modelling framework.

**- 03_Pre-processing.R:** Performs data wrangling and generates a clean data set of observed and foreacsted dengue 
and ancillary data.

**- 04_Fit_models.R:** Fits all probabilistic models required to generate the superensemble, and creates a superensemble
of probabilistic dengue models using an Integrated nested Laplace approximations (INLA) approach and the Continuous 
Rank Probability Score.

**- 05_Model_outputs.R:** Generates _csv_ output files and stores them in the sub-folder _output_.

**- 06_Run_routines.R:** Runs all previous sub-routines. In this file, users will need to specify the folder and 
sub-folders where data and scripts are stored.

For any issues with the code please contact [Felipe J Colón-González](Felipe.Colon@lshtm.ac.uk)

DOI: [https://doi.org/10.5281/zenodo.4740094](https://doi.org/10.5281/zenodo.4740094)

