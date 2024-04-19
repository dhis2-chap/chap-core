#'Run EPIDEMIA forecast models and early detection algorithm.
#'
#'The Epidemic Prognosis Incorporating Disease and Environmental Monitoring for
#'Integrated Assessment (EPIDEMIA) Forecasting System is a set of tools coded in
#'free, open-access software, that integrate surveillance and environmental data
#'to model and create short-term forecasts for environmentally-mediated
#'diseases. This function, \code{epidemiar::run_epidemia()} is the central
#'function to model and forecast a wide range of environmentally-mediated
#'diseases.
#'
#'For more a longer description of the package, run the following command to see
#'the overview vignette: \code{vignette("overview-epidemiar", package =
#'"epidemiar")}
#'
#'For more details run the following command to see the vignette on input data
#'and modeling parameters: \code{vignette("data-modeling", package =
#'"epidemiar")}
#'
#'@param epi_data Epidemiological data with case numbers per week, with date
#'  field "obs_date".
#'@param env_data Daily environmental data for the same groupfields and date
#'  range as the epidemiological data. It may contain extra data (other
#'  districts or date ranges). The data must be in long format (one row for each
#'  date and environmental variable combination), and must start at absolutel
#'  minimum \code{report_settings$env_lag_length} days (default 180) before
#'  epi_data for forecasting.
#'@param env_ref_data Historical averages by week of year for environmental
#'  variables. Used in extended environmental data into the future for long
#'  forecast time, to calculate anomalies in early detection period, and to
#'  display on timeseries in reports.
#'@param env_info Lookup table for environmental data - reference creation
#'  method (e.g. sum or mean), report labels, etc.
#'
#'@param casefield The column name of the field that contains disease case
#'  counts (unquoted field name).
#'@param populationfield Column name of the optional population field to give
#'  population numbers over time (unquoted field name). Used to calculated
#'  incidence if \code{report_settings$report_value_type} = "incidence". Also
#'  optionally used in Farrington method for populationOffset.
#'@param groupfield The column name of the field for district or geographic area
#'  unit division names of epidemiological AND environmental data (unquoted
#'  field name). If there are no groupings (all one area), user should give a
#'  field that contains the same value throughout.
#'@param obsfield Field name of the environmental data variables (unquoted field
#'  name).
#'@param valuefield Field name of the value of the environmental data variable
#'  observations (unquoted field name).
#'
#'@param fc_model_family The \code{\link[stats]{family}} parameter passsed to
#'  \code{\link[mgcv:bam]{mgcv::bam}}, and the extended families in
#'  \code{\link[mgcv]{family.mgcv}} can also be used. This sets the type of
#'  generalized additive model (GAM) to run: it specifies the distribution and
#'  link to use in model fitting. E.g. for a Poisson regression, the user would
#'  input "poisson()". If a cached model is being used, set the parameter to
#'  `"cached"`.
#'
#'@param report_settings This is a named list of all the report, forecasting,
#'  event detection and other settings. All of these have defaults, but they are
#'  not likely the defaults needed for your system, so each of these should be
#'  reviewed:
#'
#'  \itemize{
#'
#'  \item \code{report_period} = 26: The number of weeks that the entire report
#'  will cover. The \code{report_period} minus \code{fc_future_period} is the
#'  number of weeks of past (known) data that will be included. Default is 26
#'  weeks.
#'
#'  \item \code{report_value_type} = "cases": How to report the results, either
#'  in terms of "cases" (default) or "incidence".
#'
#'  \item \code{report_inc_per} = 1000: If reporting incidence, what should be
#'  denominator be?  Default is per 1000 persons.
#'
#'  \item \code{epi_date_type} = "weekISO": String indicating the standard (WHO
#'  ISO-8601 or CDC epi weeks) that the weeks of the year in epidemiological and
#'  environmental reference data use ("weekISO" or "weekCDC"). Required:
#'  epidemiological observation dates listed are LAST day of week.
#'
#'  \item \code{epi_interpolate} = FALSE: TRUE/FALSE flag for if the given
#'  epidemiological data be linearly interpolated for any explicitly missing
#'  values before modeling?
#'
#'  \item \code{epi_transform} = "none" (default if not set): Should the case
#'  counts be transformed just before regression modeling and backtransformed
#'  directly after prediction/forecast creation? The current only supported
#'  transformation is "log_plus_one", where log(cases + 1) is modeled and
#'  back-transformed by exp(pred) - 1 (though pmax(exp(pred) - 1, 0) is used in
#'  case of small predicted values).
#'
#'  \item \code{model_run} = FALSE: TRUE/FALSE flag for whether to only generate
#'  the model regression object plus metadata. This model can be cached and used
#'  later on its own, skipping a large portion of the slow calculations for
#'  future runs.
#'
#'  \item \code{model_cached} = NULL: The output of a previous model_run = TRUE
#'  run of run_epidemia() that produces a model (regression object) and
#'  metadata. The metadata will be used for input checking and validation. Using
#'  a prebuilt model saves on processing time, but will need to be updated
#'  periodically. If using a cached model, also set `fc_model_family =
#'  "cached"`.
#'
#'  \item \code{env_var}: List environmental variables to actually use in the
#'  modelling. (You can therefore have extra variables or data in the
#'  environmental dataset.) Input should be a one column tibble, header row as
#'  `obsfield` and each row with entries of the variables (must match what is in
#'  env_data, env_ref-data, and env_info). Default is to use all environmental
#'  variables that are present in all three of env_data, env_ref_data, and
#'  env_info.
#'
#'  \item \code{env_lag_length} = 181: The number of days of past environmental
#'  data to include for the lagged effects. The distributed lags are summarized
#'  using a thin plate basis function. Default is 181 days.
#'
#'  \item \code{env_anomalies} = FALSE: TRUE/FALSE indicating if the
#'  environmental variables should be replaced with their anomalies. The
#'  variables were transformed by taking the residuals from a GAM with
#'  geographic unit and cyclical cubic regression spline on day of year per
#'  geographic group.
#'
#'  \item \code{fc_start_date}: The date to start the forecasting, also the
#'  start of the early warning period. Epidemiological data does not have to
#'  exist just before the start date, though higher accuracy will be obtained
#'  with more recent data. The default is the week following the last known
#'  observation in /code{epi_data}.
#'
#'  \item \code{fc_future_period} = 8: Number of future weeks from the end of
#'  the \code{epi_data} to produce forecasts, or if fc_start_date is set, the
#'  number of weeks from and including the start date to create forecasts.
#'  Synonymous with early warning period. Default is 8 weeks.
#'
#'  \item \code{fc_clusters}: Dataframe/tible of geographic units and a cluster
#'  id. This clusters, or groups, certain geographic locations together, to
#'  better model when spatial non-stationarity in the relationship between
#'  environmental variables and cases. See the overview and data & mdoeling
#'  vignettes for more discussion. Default is a global model, all geographic
#'  units in one cluster.
#'
#'  \item \code{fc_cyclicals} = FALSE: TRUE/FALSE flag on whether to include a
#'  smooth term based on day of year in the modeling (as one way of accounting
#'  for seasonality).
#'
#'  \item \code{fc_cyclicals_by}: Unit to run the `fc_cyclicals` terms by.
#'  Either by 'cluster' (default; clusters given by `fc_clusters`) or by 'group'
#'  (per geogroup in `groupfield`).
#'
#'  \item \code{fc_splines}: The type of splines that will be used to handle
#'  long-term trends and lagged environmental variables. If supplemental package
#'  `clusterapply` is not installed, the default (and only choice) uses modified
#'  b-splines ('modbs'). If the package is installed, then 'tp' becomes an
#'  option and the default which uses thin plate splines instead.
#'
#'  \item \code{fc_ncores}: The number of physical CPU cores available. Will be
#'  used to determine the multi-threading (or not) for use in modeling and
#'  predicting.
#'
#'  \item \code{ed_summary_period} = 4: The number of weeks that will be
#'  considered the "early detection period". It will count back from the week of
#'  last known epidemiological data. Default is 4 weeks.
#'
#'  \item \code{ed_method} = 'none': Which method for early detection should be
#'  used ("farrington" is only current option, or "none").
#'
#'  \item \code{ed_control} = Controls passed along to the event detection
#'  method.  E.g. for `ed_method = 'farrington'`, these are passed to
#'  \code{\link[surveillance:farringtonFlexible]{surveillance::farringtonFlexible()}}.
#'   Currently, these parameters are supported for Farrington: `b`, `w`,
#'  `reweight`, `weightsThreshold`, `trend`, `pThresholdTrend`,
#'  `populationOffset`, `noPeriods`, `pastWeeksNotIncluded`, `thresholdMethod`.
#'  Any control not included will use surveillance package defaults, with the
#'  exception of `b`, the number of past years to include: epidemiar default is
#'  to use as many years are available in the data.
#'
#'
#'  }
#'
#'@return Returns a suite of summary and report data.
#'
#'  1. \code{summary_data}: Early detection and early warning alerts levels for
#'  each woreda. Early detection alerts (ed_alert_count) are alerts that are
#'  triggered during the early detection period, which is defined as the 4 most
#'  recent weeks of known epidemiology data. Similarly, early warning alerts
#'  (ew_alert_count) are alerts in the future forecast estimates. “High” level
#'  indicates two or more weeks in this period had incidences greater than the
#'  alert threshold, “Medium” means that one week was in alert status, and “Low”
#'  means no weeks had alerts (ed_sum_level and ew_level, respectively).
#'
#'  2. \code{epi_summary}: Mean disease incidence per geographic group during
#'  the early detection period.
#'
#'  3. \code{modeling_results_data}:These are multiple timeseries values for
#'  observed, forecast, and alert thresholds of disease incidence, over the
#'  report period, for each geographic unit. These data can be used in creating
#'  the individual geographic unit control charts.
#'
#'  4. \code{environ_timeseries}: These are multiple timeseries for the
#'  environmental variables during the report period for each geographic unit.
#'
#'  5. \code{environ_anomalies}: These data are the recent (during the early
#'  detection period) differences (anomalies) of the environmental variable
#'  values from the climatology/reference mean.
#'
#'  6. \code{params_meta}: This lists dates, settings, and parameters that
#'  \code{run_epidemiar()} was called with.
#'
#'  7. \code{regression_object}: This is the regression object from the general
#'  additive model (GAM, parallelized with BAM) regression. This is only for
#'  statistical investigation of the model, and is usually not saved (very large
#'  object).
#'
#'  For more details see the vignette on the output data:
#'  \code{vignette("output-report-data", package = "epidemiar")}
#'
#'  However, if \code{model_run = TRUE}, the function returns a list of two
#'  objects. The first, \code{model_obj} is the regression object from whichever
#'  model is being run. There is also \code{model_info} which has details on the
#'  parameters used to create the model, similar to \code{params_meta} in a full
#'  run.
#'
#'@examples "See model_forecast_script in epidemiar-demo for full example:
#'https://github.com/EcoGRAPH/epidemiar-demo"
#'
#'@export
#'
#'@importFrom magrittr %>%
#'@importFrom rlang !!
#'@importFrom rlang :=
#'@importFrom rlang .data
#'@importFrom lubridate %within%


## Main Modeling (Early Detection, Forecasting) Function
run_epidemia <- function(epi_data = NULL,
                         env_data = NULL,
                         env_ref_data = NULL,
                         env_info = NULL,
                         #fields
                         casefield = NULL,
                         groupfield = NULL,
                         populationfield = NULL,
                         obsfield = NULL,
                         valuefield = NULL,
                         #required settings
                         fc_model_family = NULL,
                         #optional
                         report_settings = NULL)
{

  #Note for model family
  # need to figure out how to handle naive models


  # For validation runs, special escapes ------------------------------------
  valid_run <-  FALSE
  calling_check <- as.list(sys.call(-1))
  #print(calling_check)
  if (length(calling_check) > 0){
    calling_function <- as.list(sys.call(-1))[[1]]
    #print(calling_function)
  } else {calling_function <- "directly"}
  if(calling_function == "run_validation" | calling_function == "epidemiar::run_validation"){
    valid_run <-  TRUE
    #message("Running model validation...")
    #rename already enquo'd variables
    quo_casefield <- casefield
    quo_popfield <- populationfield
    quo_groupfield <- groupfield
    quo_obsfield <- obsfield
    quo_valuefield <- valuefield
  }

  # Non-standard evaluation quosures ----------------------------------------

  #if from run_validation, fields are already quosures, so skip
  if(!valid_run){
    # dplyr programming steps for passing of field names
    quo_casefield <- rlang::enquo(casefield)
    quo_popfield <- rlang::enquo(populationfield)
    quo_groupfield <- rlang::enquo(groupfield)
    quo_obsfield <- rlang::enquo(obsfield)
    quo_valuefield <- rlang::enquo(valuefield)

    #Note: if field name does not exist in any dataset, enquo() will throw an error.
  }

  # Preparing: Basic Input checking -----------------------------------------------

  #First: Test for critical inputs. This will not check if they've assigned the right
  #thing to the argument, or got the argument order correct if not explicit
  #argument declarations. But, no other checks can really proceed if things are
  #missing.

  #NSE is a little tricky: can't test directly on fields-to-be-enquo'd because
  #it'll try to evaluate them, and complain that the object (actually field
  #name) doesn't exist.

  #looping along lists gets difficult/impossible when things might be missing,
  # so testing each individually.

  #initialize missing info msgs & flag
  missing_msgs <- ""
  missing_flag <- FALSE

  #NSE fields
  if (rlang::quo_is_null(quo_casefield)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "casefield", sep = "\n")
  }
  if (rlang::quo_is_null(quo_groupfield)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "groupfield", sep = "\n")
  }
  if (rlang::quo_is_null(quo_obsfield)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "obsfield", sep = "\n")
  }
  if (rlang::quo_is_null(quo_valuefield)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "valuefield", sep = "\n")
  }
  #note: population can be missing (case based reports, not incidence)

  #data & model form
  if (is.null(epi_data)){
        missing_flag <- TRUE
        missing_msgs <- paste0(missing_msgs, "epi_data", sep = "\n")
  }
  if (is.null(env_data)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "env_data", sep = "\n")
  }
  if (is.null(env_ref_data)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "env_ref_data", sep = "\n")
  }
  if (is.null(env_info)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "env_info", sep = "\n")
  }
  if (is.null(fc_model_family)){
    missing_flag <- TRUE
    missing_msgs <- paste0(missing_msgs, "fc_model_family", sep = "\n")
  }
  #if missing, stop and give error message
  if (missing_flag){
    stop("Missing critical argument(s). Please make sure the following is/are included:\n", missing_msgs)
  }


  # Preparing: generating listings, detailed input checks, defaults ----------------------------

  #create alphabetical list of unique groups
  #must remain in alpha order for early detection using surveillance package to capture results properly
  groupings <- dplyr::pull(epi_data, !!quo_groupfield) %>% unique() %>% sort()
  #create alphabetical list of all unique environmental variables in env_data
  env_variables <- dplyr::pull(env_data, !!quo_obsfield) %>% unique() %>% sort()


  # Detailed input checking and sets defaults in report_settings if not supplied
  check_results <- input_check(epi_data,
                               env_data,
                               env_ref_data,
                               env_info,
                               quo_casefield,
                               quo_popfield,
                               quo_groupfield,
                               quo_obsfield,
                               quo_valuefield,
                               fc_model_family,
                               raw_settings = report_settings,
                               groupings,
                               env_variables)
  #if warnings, just give message and continue
  if (check_results$warn_flag){
    message(check_results$warn_msgs)
  }
  #if then if errors, stop and return error messages
  if (check_results$err_flag){
    #prevent possible truncation of all error messages
    options(warning.length = 4000L)
    stop(check_results$err_msgs)
  }

  #update report_settings with checked, cleaned, or newly added default values
  report_settings <- check_results$clean_settings

  # switch epi_date_type to week_type needed for add_datefields()
  week_type <- dplyr::case_when(
    report_settings[["epi_date_type"]] == "weekISO" ~ "ISO",
    report_settings[["epi_date_type"]] == "weekCDC"  ~ "CDC",
    #default NA
    TRUE             ~ NA_character_)



  # Preparing: date sets ----------------------------

  # Some additional checks and overrides now that the others have been done
  #is the user given date the end date of a epidemiolgical week?
  # calculate expected date and compare
  #switch on ISO/CDC weeks
  if(report_settings[["epi_date_type"]] == "weekISO"){
    user_st_year <- lubridate::isoyear(report_settings[["fc_start_date"]])
    user_st_week <- lubridate::isoweek(report_settings[["fc_start_date"]])
    expected_date <- make_date_yw(year = user_st_year,
                                  week = user_st_week,
                                  weekday = 7,
                                  system = "ISO")
  } else {
    #can add more if blocks later for other date types
    #"weekCDC"
    user_st_year <- lubridate::epiyear(report_settings[["fc_start_date"]])
    user_st_week <- lubridate::epiweek(report_settings[["fc_start_date"]])
    expected_date <- make_date_yw(year = user_st_year,
                                  week = user_st_week,
                                  weekday = 7,
                                  system = "CDC")
  }
  #if not, override and provide message
  if(!report_settings[["fc_start_date"]] == expected_date){
    message("Note: 'report_settings$fc_start_date was not the end date of an epidemiological week\n
            Using ", expected_date, " instead.")
    report_settings[["fc_start_date"]] <- expected_date
  }


  # Create report date information: for passing to interval functions, and report output
  # report_period is full # of weeks of report.
  # fc_future_period is how many of those weeks should be in the future.
  # *Nearly all dates are calculated from fc_start_date*
  #     Forecast begins on fc_start_date, runs for fc_future_period
  #     Report_period minus fc_future_period is the number of 'past' weeks to include
  #     Known data is independent of fc_start_date but important for early detection

  #time units #report_settings[["epi_date_type"]]
  #as.difftime cannot do monthly, so will have to build switch for different function calculations

  #full report
  report_dates <- list(full = list(min = report_settings[["fc_start_date"]] -
                                     lubridate::as.difftime((report_settings[["report_period"]] -
                                                               report_settings[["fc_future_period"]]),
                                                            unit = "weeks"),
                                   max = report_settings[["fc_start_date"]] +
                                     lubridate::as.difftime((report_settings[["fc_future_period"]] - 1),
                                                            units = "weeks")))
  report_dates$full$seq <- report_dates$full %>% {seq.Date(.$min, .$max, "week")}

  #dates with known epidemological data (note: may NOT be in report period)
  report_dates$known <- list(min = min(epi_data$obs_date, na.rm = TRUE),
                             max = max(epi_data$obs_date, na.rm = TRUE))
    #can't assume known is complete sequence now
    #report_dates$known$seq <- report_dates$known %>% {seq.Date(.$min, .$max, "week")}
    #any known data in range (across any geographical groupings)
  report_dates$known$seq <- epi_data %>% dplyr::pull(.data$obs_date) %>% unique() %>% sort()

  #forecast period
  report_dates$forecast <- list(min = report_settings[["fc_start_date"]],
                                #could calculate from forecast_future, but already done so in $full
                                max = report_dates$full$max)
  report_dates$forecast$seq <- report_dates$forecast %>% {seq.Date(.$min, .$max, "week")}

  #early detection summary period (ED runs over full report, this is for defined early DETECTION period)
  if (report_settings[["ed_summary_period"]] > 0) {
    report_dates$ed_sum <- list(min = report_settings[["fc_start_date"]] -
                                  lubridate::as.difftime(report_settings[["ed_summary_period"]],
                                                         units = "weeks"),
                                max = report_settings[["fc_start_date"]] -
                                  lubridate::as.difftime(1, units = "weeks"))
    report_dates$ed_sum$seq <- report_dates$ed_sum %>% {seq.Date(.$min, .$max, "week")}
  } else {
    #no early detection period (ed_summary_period = 0 or weird)
    report_dates$ed_sum <- list(min = NA, max = NA, seq = NA)
  }

  #period of report NOT in forecast ("previous" to forecast)
  report_dates$prev <- list(min = report_dates$full$min,
                            max = report_settings[["fc_start_date"]] -
                              lubridate::as.difftime(1, units = "weeks"))
  report_dates$prev$seq <- report_dates$prev %>% {seq.Date(.$min, .$max, "week")}



  # Preparing: data checks for implicit missing, NA and interpolation ---------------------

  # Implicit missing, or gaps introduced by user start parameter, may exist
  # implicit missing may also exist in historical/known time ranges
  # NOT in forecast period, as that will be handled by 'future' extension
  epi_all_dates <- seq.Date(from = report_dates$known$min, to = report_dates$prev$max, by = "week")

  epi_full <- tidyr::crossing(obs_date = epi_all_dates, #report_dates$prev$seq,
                              group_temp = groupings)
  #and fix names with NSE
  epi_full <- epi_full %>%
    dplyr::rename(!!rlang::as_name(quo_groupfield) := .data$group_temp)

  #antijoin with existing data to find rows are implicitly missing
  epi_implicit <- epi_full %>%
    dplyr::anti_join(epi_data, by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                                       "obs_date"),
                                                     c(rlang::as_name(quo_groupfield),
                                                       "obs_date")))
  #bind missing
  epi_data <- epi_data %>%
    dplyr::bind_rows(epi_implicit) %>%
    #and sort by alphabetical groupfield and date
    dplyr::arrange(!!quo_groupfield, .data$obs_date)

  #fill down for population that we would need values, if pop field present
  #rest will remain NA for implicit missing data: casefield, and any extra/extraneous columns in original data
  if(!rlang::quo_is_null(quo_popfield)){
    epi_data <- epi_data %>%
      #per geographic group
      dplyr::group_by(!!quo_groupfield) %>%
      #fill population down ('persistence' fill, last known value carried forward)
      tidyr::fill(!!quo_popfield, .direction = "down") %>%
      #ungroup to finish
      dplyr::ungroup()
  }


  #Interpolate NAs if user selected
  if (report_settings[["epi_interpolate"]] == TRUE){
    #Note: cases_epidemiar is field name returned (epi)
    epi_data <- epi_NA_interpolate(epi_data, quo_casefield, quo_groupfield) %>%
      #and sort by alphabetical groupfield (dates should already be sorted from interpolate function)
      dplyr::arrange(!!quo_groupfield, .data$obs_date)
  } else {
    epi_data <- epi_data %>%
      #copy over value
      dplyr::mutate(cases_epidemiar = !!quo_casefield) %>%
      #force into integer, just in case/keeping consistency
      dplyr::mutate(cases_epidemiar = floor(.data$cases_epidemiar)) %>%
      #and sort by alphabetical groupfield, dates
      dplyr::arrange(!!quo_groupfield, .data$obs_date)
  }


    #Note: val_epidemiar is field name (env)
  #prep environmental data, filling in of missing data will happen in extend_env_future()
  env_data <- env_data %>%
    #copy over value
    dplyr::mutate(val_epidemiar = !!quo_valuefield)


  # Set up output report data format ----------------------------------------

  #create observed data series
  obs_res <- epi_data %>%
    #include only observed data from during report period
    dplyr::filter(.data$obs_date %in% report_dates$full$seq) %>%
    dplyr::mutate(series = "obs",
                  #value calculations change depending on report_value_type
                  #case_when is not viable because it evaluates ALL RHS
                  #condition is scalar, so vectorized ifelse is not appropriate
                  value = if(report_settings[["report_value_type"]] == "cases"){
                            .data$cases_epidemiar
                          } else if (report_settings[["report_value_type"]] == "incidence"){
                            .data$cases_epidemiar / !!quo_popfield * report_settings[["report_inc_per"]]
                          } else {NA_real_},
                  #note: uses cases_epidemiar, so will return interpolated values if user selected interpolation
                  lab = "Observed",
                  upper = NA,
                  lower = NA) %>%
    dplyr::select(!!quo_groupfield, .data$obs_date, .data$series, .data$value,
                  .data$lab, .data$upper, .data$lower)



  # Forecasting -------------------------------------------------------------

  fc_res_all <- run_forecast(epi_data,
                             quo_popfield,
                             quo_groupfield,
                             env_data,
                             quo_obsfield,
                             quo_valuefield,
                             env_ref_data,
                             env_info,
                             fc_model_family,
                             report_settings,
                             #internal/calculated
                             valid_run,
                             groupings,
                             env_variables,
                             report_dates)


  #if we are only generating the model, then end here
  if (report_settings[["model_run"]]){
    message("Model run only, returning regression object and model information.")

    fieldnames <- list(casefield = rlang::as_name(quo_casefield),
                       populationfield = rlang::as_name(quo_popfield),
                       groupfield = rlang::as_name(quo_groupfield),
                       obsfield = rlang::as_name(quo_obsfield),
                       valuefield = rlang::as_name(quo_valuefield))


    model_meta <- create_named_list(date_created = Sys.Date(),
                                    fieldnames,
                                    groupings,
                                    fc_model_family,
                                    env_variables_used = fc_res_all$env_variables_used,
                                    env_dt_ranges = fc_res_all$env_dt_ranges,
                                    known_epi_range = report_dates$known,
                                    env_info,
                                    report_settings = format_report_settings(report_settings),
                                    date_created = Sys.Date())


    #if a model run, forecast result contains regression object
    model_results <- list(model_obj = fc_res_all$reg_obj,
                          model_info = model_meta)

    return(model_results)

  }



  # Event detection ---------------------------------------------------------

  #need to calculate event detection on observed data; & in forecast period, the FORECASTED results

  #however, to force surveillance into giving a threshold even when input is NA, use forecast values if NA
  # but will need to censor those dates from alerts later

  if (report_settings[["ed_method"]] == "farrington") {

    #need to include farrington spin up period (from limit54 parameter, default 4 time units),
    # add buffer before prev period starts
    # Get user values from limit54 control
    if (is.null(report_settings[["ed_control"]][["limit54"]])){
      #default is limit54 = c(5,4): 4 time units
      far_buffer <- 4 + 1
    } else {
      #get user value
      far_buffer <- report_settings[["ed_control"]][["limit54"]][[2]] + 1
    }

    #note dev_fc_fit_freq 'week' fits are report period only, and will not have this buffer period (NA)
    far_buffer_start_date <- report_dates$prev$min - lubridate::as.difftime(far_buffer,
                                                                            units = "weeks")

    #existing data before report start + buffer
    # (i.e. before we have any modelled values from forecasting)
    epi_to_fc <- epi_data %>%
      dplyr::filter(.data$obs_date < far_buffer_start_date)

    #dates that we need to blend obs/fc
    far_blend_dates <- seq.Date(from = far_buffer_start_date,
                          to = report_dates$prev$max,
                          by = "week")

    #observed OR modeled values in report period before forecasting ('previous')
    report_prev_values <- fc_res_all$fc_epi %>%
      #get results ONLY from prev period + buffer
      dplyr::filter(.data$obs_date %in% far_blend_dates) %>%
      #flag dates that will need to be censored later
      dplyr::mutate(censor_flag = rlang::are_na(.data$cases_epidemiar),
                    #and fill in NA values for modelled values for continuous non-NA values
                    cases_epidemiar = ifelse(!rlang::are_na(.data$cases_epidemiar),
                                            .data$cases_epidemiar,
                                            .data$fc_cases))

    #modeled values in forecast period = forecast values
    forecast_values <- fc_res_all$fc_epi %>%
      #get future forecasted results ONLY
      dplyr::filter(.data$obs_date %in% report_dates$forecast$seq) %>%
      #assign forecasted values into cases_epidemiar column (so event detection will run on these values)
      dplyr::mutate(cases_epidemiar = .data$fc_cases)


    #combine all
    obs_fc_epi <- dplyr::bind_rows(epi_to_fc, report_prev_values, forecast_values) %>%
      #will be lost by end, but need for event detection methods using surveillance::sts objects
      epidemiar::add_datefields() %>%
      #arrange (for viewing/checking)
      dplyr::arrange(!!quo_groupfield, .data$obs_date)



  } else {
    #normal combination of past and forecasted values
    #existing data before forecast start
    epi_to_fc <- epi_data %>%
      dplyr::filter(.data$obs_date < report_dates$forecast$min)

    #modeled values in forecast period = forecast values
    forecast_values <- fc_res_all$fc_epi %>%
      #get future forecasted results ONLY
      dplyr::filter(.data$obs_date %in% report_dates$forecast$seq) %>%
      #assign forecasted values into cases_epidemiar column (so event detection will run on these values)
      dplyr::mutate(cases_epidemiar = .data$fc_cases)

    #combine existing and future
    obs_fc_epi <- dplyr::bind_rows(epi_to_fc, forecast_values) %>%
      #will be lost by end, but need for event detection methods using surveillance::sts objects
      epidemiar::add_datefields() %>%
      #arrange (for viewing/checking)
      dplyr::arrange(!!quo_groupfield, .data$obs_date)

  }


  #run event detection on combined dataset
  ed_res <- run_event_detection(epi_fc_data = obs_fc_epi,
                                quo_groupfield,
                                quo_popfield,
                                ed_method = report_settings[["ed_method"]],
                                ed_control = report_settings[["ed_control"]],
                                val_type = report_settings[["report_value_type"]],
                                inc_per = report_settings[["report_inc_per"]],
                                groupings,
                                report_dates,
                                valid_run)


  # Combining forecast and event detection results --------------------------

  ## Combine epi datasets
  epi_res <- dplyr::bind_rows(obs_res, fc_res_all$fc_res, ed_res)
  #add week fields
  modeling_results_data <- epidemiar::add_datefields(epi_res, week_type)


  # Format other data for report --------------------------------------------

  #Do all the other data and reporting if not a validation run
  if(!valid_run){

    ## Prep Environmental Data for report
    #using extended environmental data from forecast functions
    environ_timeseries <- environ_report_format(env_ext_data = fc_res_all$env_data_extd,
                                                env_ref_data,
                                                quo_groupfield,
                                                quo_obsfield,
                                                env_used = fc_res_all$env_variables_used,
                                                env_info,
                                                epi_date_type = report_settings[["epi_date_type"]],
                                                report_dates)

    ##Environmental Anomaly Data (during ED period)
    environ_anomalies <- calc_env_anomalies(env_ts = environ_timeseries,
                                            quo_groupfield,
                                            quo_obsfield,
                                            report_dates)

    ## Calculate Early Detection and Early Warning summary data
    summary_data <- create_summary_data(ed_res,
                                        quo_groupfield,
                                        report_dates)

    ## Calculate epidemiological (incidence) summary
    epi_summary <- create_epi_summary(obs_res,
                                      quo_groupfield,
                                      report_dates)

    ## Parameters and metadata that might be useful in report generation
    # all of these may not be needed
    fieldnames <- list(casefield = rlang::as_name(quo_casefield),
                       populationfield = rlang::as_name(quo_popfield),
                       groupfield = rlang::as_name(quo_groupfield),
                       obsfield = rlang::as_name(quo_obsfield),
                       valuefield = rlang::as_name(quo_valuefield))

    params_meta <- create_named_list(date_created = Sys.Date(),
                                     fieldnames,
                                     groupings,
                                     fc_model_family,
                                     env_variables_used = fc_res_all$env_variables_used,
                                     env_dt_ranges = fc_res_all$env_dt_ranges,
                                     report_dates,
                                     env_info,
                                     report_settings = format_report_settings(report_settings))

    #regression object for future other use or troubleshooting
    regression_object <- fc_res_all$reg_obj


    # Collect all results -----------------------------------------------------

    all_results <- create_named_list(summary_data,
                                     epi_summary,
                                     modeling_results_data,
                                     environ_timeseries,
                                     environ_anomalies,
                                     params_meta,
                                     regression_object)
    message("Finished.")
    return(all_results)

    # end !valid run
  } else {
    ## Validation runs needs only small subset of data

      valid_mod_results <- create_named_list(modeling_results_data)

      message("Finished week of validation run.")

      return(valid_mod_results)
    } #end else of !valid run


} #end run_epidemia() function
