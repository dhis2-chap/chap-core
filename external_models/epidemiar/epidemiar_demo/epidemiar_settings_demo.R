# ###############################################################################################
#
# External file that contains all the settings for generating an epidemiar forecast
#
# Sections:
# 1. General report and epidemiological parameters
# 2. Environmental parameters
# 3. Forecasting/Modeling parameters
# 4. Event Detection parameters
#
# Settings are for the Amhara region of Ethiopia
#
# ###############################################################################################

## IMPORTANT: If you add/remove parameters - 
#   make sure to change the report_settings list creation at the bottom

# 1. Set up general report and epidemiological parameters ----------

#total number of weeks in report (including forecast period)
report_period <- 26

#report out in incidence 
report_value_type <-  "incidence"

#report incidence rates per 1000 people
report_inc_per <- 1000

#date type in epidemiological data
epi_date_type <- "weekISO"

#interpolate epi data?
epi_interpolate <- TRUE

#use a transformation on the epi data for modeling? ("none" if not)
#Note that this is closely tied with the model family parameter below
#   fc_model_family <- "gaussian()"
epi_transform <- "log_plus_one"

#model runs and objects
model_run <- FALSE
model_cached <- NULL


# 2. Set up environmental vars ------------------------------------

#read in model environmental variables to use
pfm_env_var <- readr::read_csv("data/falciparum_model_envvars.csv", col_types = readr::cols())
pv_env_var <- readr::read_csv("data/vivax_model_envvars.csv", col_types = readr::cols())

#set maximum environmental lag length (in days)
env_lag_length <- 181

#use environmental anomalies for modeling?
# TRUE for poisson model
env_anomalies <- TRUE


# 3. Set up forecast controls -------------------------------------

#Model choice and parameters
#Note that this is closely tied with the epi_transform <- "log_plus_one"
# parameter in the report and epidemiological parameter settings
fc_model_family <- "gaussian()"

#Spline choice for long-term trend and lagged environmental variables
fc_splines <- "modbs" #modified b-splines, faster but not as good as thin plate
#fc_splines <- "tp" #requires clusterapply companion package

#Include seasonal cyclical in modeling? 
fc_cyclicals <- TRUE

#forecast 8 weeks into the future
fc_future_period <- 8

#read in model cluster information
pfm_fc_clusters <- readr::read_csv("data/falciparum_model_clusters.csv", col_types = readr::cols())
pv_fc_clusters <- readr::read_csv("data/vivax_model_clusters.csv", col_types = readr::cols())

#info for parallel processing on the machine the script is running on
fc_ncores <- max(parallel::detectCores(logical=FALSE),
                 1,
                 na.rm = TRUE)


# 4. Set up early detection controls -------------------------------

#number of weeks in early detection period (last n weeks of known epidemiological data to summarize alerts)
ed_summary_period <- 4

#event detection algorithm
ed_method <- "Farrington"

#settings for Farrington event detection algorithm
pfm_ed_control <- list(
  w = 3, reweight = TRUE, weightsThreshold = 2.58,
  trend = TRUE, pThresholdTrend = 0,
  populationOffset = TRUE,
  noPeriods = 12, pastWeeksNotIncluded = 4,
  limit54=c(1,4), 
  thresholdMethod = "nbPlugin")

pv_ed_control <- list(
  w = 4, reweight = TRUE, weightsThreshold = 2.58,
  trend = TRUE, pThresholdTrend = 0,
  populationOffset = TRUE,
  noPeriods = 10, pastWeeksNotIncluded = 4,
  limit54 = c(1,4), 
  thresholdMethod = "nbPlugin")


# Combine report settings -------------------------------

pfm_report_settings <- epidemiar::create_named_list(report_period,
                                                    report_value_type,
                                                    report_inc_per,
                                                    epi_date_type,
                                                    epi_interpolate,
                                                    epi_transform,
                                                    model_run,
                                                    env_var = pfm_env_var,
                                                    env_lag_length,
                                                    env_anomalies,
                                                    fc_splines,
                                                    fc_cyclicals,
                                                    fc_future_period,
                                                    fc_clusters = pfm_fc_clusters,
                                                    fc_ncores,
                                                    ed_summary_period,
                                                    ed_method,
                                                    ed_control = pfm_ed_control)

pv_report_settings <- epidemiar::create_named_list(report_period,
                                                   report_value_type,
                                                   report_inc_per,
                                                   epi_date_type,
                                                   epi_interpolate, 
                                                   epi_transform,
                                                   model_run,
                                                   env_var = pv_env_var,
                                                   env_lag_length,
                                                   env_anomalies,
                                                   fc_splines,
                                                   fc_cyclicals,
                                                   fc_future_period,
                                                   fc_clusters = pv_fc_clusters,
                                                   fc_ncores,
                                                   ed_summary_period,
                                                   ed_method,
                                                   ed_control = pv_ed_control)

#Note that fc_model_family set in Forecast section is a top-level parameter and separate from the rest of the report_settings.