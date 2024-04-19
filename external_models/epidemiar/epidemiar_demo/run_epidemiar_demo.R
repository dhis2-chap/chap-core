# ###############################################################################################
#
# This script does all the data collating and processing before calling the epidemiar function
# to run the modeling, forecasting, and early detection and early warning alert algorithms. 
#
# At the end, this script will generate a pdf report using epidemia_report_demo.Rnw
#
# See documentation/walkthrough.pdf for more details on each step. 
#
# ###############################################################################################


# 1. Libraries & Functions ------------------------------------------------------

## Load packages necessary for script

#make sure pacman is installed
options(repos = c(CRAN = "https://cran.r-project.org"))

install.packages("pacman")
install.packages("remotes")
remotes::install_github("EcoGRAPH/epidemiar")
library(pacman)


if (!require("pacman")) install.packages("pacman")

#load packages
  #note: data corrals use dplyr, tidyr, lubridate, readr, readxl, epidemiar
  #note: pdf creation requires knitr, tinytex
pacman::p_load(dplyr,
               #knitr,
               lubridate,
               parallel,
               readr, 
               readxl,
               tidyr,
               tinytex,
               tools)
#load specialized package (https://github.com/EcoGRAPH/epidemiar)
library(epidemiar)

## Locally-defined Functions
source("R/data_corrals.R") 
source("R/report_save_create_helpers.R")

#due to experimental dplyr::summarise() parameter
options(dplyr.summarise.inform=F)

# 2. Reading in the Data -----------------------------------------------------

# read in woreda metadata
report_woredas <- read_csv("data/amhara_woredas.csv") %>% 
  filter(report == 1)
#Note: report woredas must have sufficient epi data, env data, and model cluster information, in appropriate files

# read & process case data needed for report
epi_data <- corral_epidemiological(report_woreda_names = report_woredas$woreda_name)

# read & process environmental data for woredas in report
env_data <- corral_environment(report_woredas = report_woredas)

## Optional: For slight speed increase, 
# date filtering to remove older environmental data.
# older env data was included to demo epidemiar::env_daily_to_ref() function.
# in make_date_yw() weekday is always end of the week, 7th day
env_start_date <- epidemiar::make_date_yw(year = 2012, week = 1, weekday = 7) 
#filter data
env_data <- env_data %>% filter(obs_date >= env_start_date)
#force garbage collection to free up memory
invisible(gc())

# read in climatology / environmental reference data
env_ref_data <- read_csv("data/env_ref_data_2002_2018.csv", col_types = cols())

# read in environmental info file
env_info <- read_xlsx("data/environ_info.xlsx", na = "NA")

# read in forecast and event detection parameters
source("data/epidemiar_settings_demo.R")

# ## OPTIONAL: Date Filtering for running certain (past) week's report
# req_date <- epidemiar::make_date_yw(year = 2016, week = 24, weekday = 7) #week is always end of the week, 7th day
# epi_data <- epi_data %>%
#   filter(obs_date <= req_date)
# env_data <- env_data %>%
#   filter(obs_date <= req_date)

# ## OPTIONAL: If instead the forecast should be beyond known epi data,
# then you can set the forecast start date directly
# pfm_report_settings$fc_start_date <- epidemiar::make_date_yw(year = 2020, week = 4, weekday = 7)
# pv_report_settings$fc_start_date <- epidemiar::make_date_yw(2020, week = 4, weekday = 7)

# # OPTIONAL: If you have created cached models to use instead of generating a new model:
# # selects the model per species with latest file created time
# # pfm
# all_pfm_models <- file.info(list.files("data/models/", full.names = TRUE, pattern="^pfm.*\\.RDS$"))
# if (nrow(all_pfm_models) > 0){
#   latest_pfm_model <- rownames(all_pfm_models)[which.max(all_pfm_models$ctime)]
#   pfm_model_cached <- readRDS(latest_pfm_model)
# }
# ##or select specific file
# #latest_pfm_model <- "data/pfm_model_xxxxxxx.RDS"
# pfm_report_settings$model_cached <- readRDS(latest_pfm_model)
# #pv
# all_pv_models <- file.info(list.files("data/models/", full.names = TRUE, pattern="^pv.*\\.RDS$"))
# if (nrow(all_pv_models) > 0){
#   latest_pv_model <- rownames(all_pv_models)[which.max(all_pv_models$ctime)]
#   pv_model_cached <- readRDS(latest_pv_model)
# }
# ##or select specific model
# #latest_pv_model <- "data/pv_model_xxxxxxxx.RDS"
# pv_report_settings$model_cached <- readRDS(latest_pv_model)



# 3. Run EPIDEMIA & create report data ---------------------------------------

#Run modeling to get report data
# with check on current epidemiology and environmental data sets

if (exists("epi_data") & exists("env_data")){
  
  # P. falciparum & mixed
  message("Running P. falciparum & mixed")
  pfm_reportdata <- run_epidemia(
    #data
    epi_data = epi_data, 
    env_data = env_data, 
    env_ref_data = env_ref_data, 
    env_info = env_info,
    #fields
    casefield = test_pf_tot, 
    groupfield = woreda_name, 
    populationfield = pop_at_risk,
    obsfield = environ_var_code, 
    valuefield = obs_value,
    #required settings
    fc_model_family = fc_model_family,
    #other settings
    report_settings = pfm_report_settings)
  
  # P. vivax
  message("Running P. vivax")
  pv_reportdata <- run_epidemia(
    #data
    epi_data = epi_data, 
    env_data = env_data, 
    env_ref_data = env_ref_data, 
    env_info = env_info,
    #fields
    casefield = test_pv_only, 
    groupfield = woreda_name, 
    populationfield = pop_at_risk,
    obsfield = environ_var_code, 
    valuefield = obs_value,
    #required settings
    fc_model_family = fc_model_family,
    #other settings
    report_settings = pv_report_settings)
  
  #if using cached models:
  #append model information to report data metadata
  if (exists('pfm_model_cached')){
    pfm_reportdata$params_meta$model_used <- latest_pfm_model
  }
  if (exists('pv_model_cached')){
    pv_reportdata$params_meta$model_used <- latest_pv_model
  }
  
} else {
  message("Error: Epidemiological and/or environmental datasets are missing.
          Check Section 2 for data error messages.")
}


# 4. Merge species data, Save, & Create PDF Report -------------------------------

if (exists("pfm_reportdata") & exists("pv_reportdata")){
  
  #merging pfm & pv data, save out, and create pdf
  merge_save_report(rpt_data_main = pfm_reportdata,
                    rpt_data_secd = pv_reportdata,
                    #mark sections as P. falciparum and mixed (pfm) or P. vivax (pv)
                    # used in the epidemia_report_demo.Rnw file for the formatted report
                    var_labs = c("pfm","pv"),
                    #save out the report data in the file that the formatting file reads
                    save_file = "report/report_data.RData",
                    #save out a second copy of the report data with year and week numbers in the name
                    second_save = TRUE,
                    #create the pdf
                    create_report = TRUE,
                    #which Rnw file to use to create pdf
                    formatting_file = "epidemia_report_demo.Rnw",
                    #append tag to file name (optional)
                    file_name_postfix = "synthetic_data",
                    #show the pdf immediately after creating
                    show_report = TRUE)
  
} else {
  message("Error: Report data for P. falciparum and/or P. vivax 
          have not been generated and are missing.")
}



# Alternative: Create pdf report ----------------------------------------------------------

# If you want to later recreate a pdf from a saved report_data file:
# Change the input report_data_file to the previously saved version
# And add a file/name for the saved output.

# create_pdf(new_data = "report/report_data_2018W52.RData",
#            #file that is loaded in the formatting file
#            report_data_file = "report/report_data.RData",
#            formatting_file = "epidemia_report_demo.Rnw",
#            #specific output file name
#            report_save_file = "report/epidemia_report_demo_2018W52.pdf",
#            show = TRUE)


