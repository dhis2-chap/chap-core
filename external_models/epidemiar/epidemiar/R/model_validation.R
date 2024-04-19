
#'Run EPIDEMIA model validation statistics
#'
#'This function takes a few more arguments than `epidemiar::run_epidemia()` to
#'generate statistics on model validation. The function will evaluate a number
#'of weeks (`total_timesteps`) starting from a specified week (`date_start`) and
#'will look at the n-week ahead forecast (1 to `timesteps_ahead` number of
#'weeks) and compare the values to the observed number of cases. An optional
#'`reporting_lag` argument will censor the last known data back that number of
#'weeks. The validation statistics include Root Mean Squared Error (RMSE) and
#'Mean Absolute Error (MAE), and an R-squared staistic both in total and per
#'geographic grouping (if present).
#'
#'@param date_start Date to start testing for model validation.
#'@param total_timesteps Number of weeks from (but including) `week_start` to
#'  run validation tests.
#'@param timesteps_ahead Number of weeks for testing the n-week ahead forecasts.
#'  Results will be generated from 1-week ahead through `weeks_ahead` number of
#'  weeks.
#'@param reporting_lag Number of timesteps to simulate reporting lag. For
#'  instance, if you have weekly data, and a reporting_lag of 1 week, and are
#'  working with a timesteps_ahead of 1 week, then that is functional equivalent
#'  to reporting lag of 0, and timesteps_ahead of 2 weeks. I.e. You are
#'  forecasting next week, but you don't know this week's data yet, you only
#'  know last week's numbers.
#'@param per_timesteps When creating a timeseries of validation results, create
#'  a moving window with per_timesteps width number of time points. Should be a
#'  minimum of 10 timesteps. In beta-testing.
#'@param skill_test Logical parameter indicating whether or not to run
#'  validations also on two naïve models for a skill test comparison. The naïve
#'  models are "persistence": the last known value (case counts) carried
#'  forward, and "average week" where the predicted value is the average of that
#'  week of the year, as calculated from historical data.
#'@param ... Accepts other arguments that may normally part of `run_epidemia()`,
#'  but ignored for validation runs.
#'
#'@inheritParams run_epidemia
#'
#'@return Returns a nested list of validation results. Statistics are calculated
#'  on the n-week ahead forecast and the actual observed case counts. Statistics
#'  returned are  Mean Absolute Error (MAE), Root Mean Squared Error (RMSE). The
#'  first object is `skill_scores`, which contains `skill_overall` and
#'  `skill_grouping`. The second list is `validations`, which contains lists per
#'  model run (the forecast model and then optionally the naive models). Within
#'  each, `validation_overall` is the results overall, `validation_grouping` is
#'  the results per geographic grouping, and `validation_perweek` is the raw
#'  stats per week. Lastly, a `metadata` list contains the important parameter
#'  settings used to run validation and when the results where generated.
#'
#'@export
#'
run_validation <- function(date_start = NULL,
                           total_timesteps = 26,
                           timesteps_ahead = 2,
                           reporting_lag = 0,
                           per_timesteps = 12,
                           skill_test = TRUE,
                           #for run_epidemia()
                           epi_data = NULL,
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
                           report_settings = NULL,
                           ...){

  #date_start: week to start reporting of results
  #total_timesteps: number of weeks forward from week_start to gather test results
  #timesteps_ahead: calculate stats on 1 to n week ahead predictions

  #this means that the start of calculations will be date_start minus timesteps_ahead # of weeks
  #then trimmed at the end to start at date_start.

  # Non-standard evaluation quosures ----------------------------------------

  # dplyr programming steps for passing of field names
  quo_casefield <- rlang::enquo(casefield)
  quo_popfield <- rlang::enquo(populationfield)
  quo_groupfield <- rlang::enquo(groupfield)
  quo_obsfield <- rlang::enquo(obsfield)
  quo_valuefield <- rlang::enquo(valuefield)

  #Note: if field name does not exist in any dataset, enquo() will throw an error.


  # Adjust parameters for validation runs -----------------------------------

  #Assumed that run_epidemia() parameters just copied and pasted, so adjust for validation
  #new lengths
  report_settings[["fc_future_period"]] <- timesteps_ahead + reporting_lag
  report_settings[["report_period"]] <- report_settings[["fc_future_period"]] + 1
  #no event detection
  report_settings[["ed_summary_period"]] <- 0 #method is 0, nothing happens
  report_settings[["ed_method"]] <- "none"
  #report out in CASES for validation
  report_settings[["report_value_type"]] <- "cases"
  #model run would make no sense here
  report_settings[["model_run"]] <- FALSE


  #for any future params accepted by run_epidemia, but are meaningless for validation runs
  # Captured, but then do nothing with them
  # Also used for hidden raw_data argument for testing/development
  # could have placed inside report_settings, but this was created first this way
  dots <- list(...)

  #Create parameter metadata
  metadata <- create_named_list(date_created = Sys.Date(),
                                date_start,
                                total_timesteps,
                                timesteps_ahead,
                                reporting_lag,
                                per_timesteps,
                                skill_test,
                                casefield = rlang::as_name(quo_casefield),
                                fc_model_family,
                                report_settings)


  # All loop prep ------------------------------------------------------

  #Set up for looping
  #preserve full data
  epi_data_orig <- epi_data
  env_data_orig <- env_data

  #Pull obs from original
  # Will have extra dates, but will be trimmed back to user requested dates later
  # May have implicit missing data, but left_joining below, so that'll create the NAs
  obs_only <- epi_data_orig %>%
    dplyr::select(!!quo_groupfield, .data$obs_date, !!quo_casefield) %>%
    #rename observation
    dplyr::rename(obs = !!rlang::as_name(quo_casefield))


  #Skill test loop set up
  if (skill_test == TRUE){
    models_to_run = c(fc_model_family, "naive-persistence", "naive-averageweek")
  } else {
    models_to_run = c(fc_model_family)
  }

  # Skill test loop ---------------------------------------------------------

  #skill test collection
  all_validations <- vector("list", length = length(models_to_run))
  #add names
  names(all_validations) <- models_to_run

  #model loop
  for (m in seq_along(models_to_run)){

    this_model <- models_to_run[m]

    #If naive-averageweek, timesteps_ahead is meaningless, just use 1
    if (this_model == "naive-averageweek"){

      this_timesteps_ahead <- 1
      this_report_settings <- report_settings
      this_report_settings[["fc_future_period"]] <- this_timesteps_ahead
      this_report_settings[["report_period"]] <- this_report_settings[["fc_future_period"]] + 1

    } else {
      #use modified fc_future_period which is timesteps_ahead + reporting_lag

      this_report_settings <- report_settings
      this_timesteps_ahead <- this_report_settings[["fc_future_period"]]

    }

    #if a naive model, drop any cached models to avoid conflicts
    if (this_model == "naive-persistence" | this_model == "naive-averageweek"){
      this_report_settings[["model_cached"]] <- NULL
    }

    # Week loop ---------------------------------------------------------------

    #Create list of dates
    #the start of calculations will be date_start minus timesteps_ahead  # of weeks
    # to total_timesteps - 1 to not count current week
    # (e.g. so total_timesteps = 52 covers 52 weeks / 1 year).
    date_list <- date_start + lubridate::weeks(-this_timesteps_ahead:(total_timesteps-1))

    #output will be list of dataframes (forecasts) until we collapse later
    fcs_list <- vector("list", length = length(date_list))

    #loop
    for (i in seq_along(date_list)){
      this_dt <- date_list[i]
      this_fc_start <- this_dt + lubridate::weeks(1)
      this_report_settings$fc_start_date <- this_fc_start

      message("Validation run: ", this_dt)

      #set up data
      #censoring as appropriate
      #reporting_lag will be handled with offset timesteps
      epi_data <- epi_data_orig %>%
        dplyr::filter(.data$obs_date <= this_dt)
      env_data <- env_data_orig %>%
        dplyr::filter(.data$obs_date <= this_dt)

      #run_epidemia
      #passing quosures, which will have an escape built into run_epidemia()
      reportdata <- run_epidemia(epi_data = epi_data,
                                 env_data = env_data,
                                 env_ref_data = env_ref_data,
                                 env_info = env_info,
                                 casefield = quo_casefield,
                                 groupfield = quo_groupfield,
                                 populationfield = quo_popfield,
                                 obsfield = quo_obsfield,
                                 valuefield = quo_valuefield,
                                 fc_model_family = this_model, #this
                                 report_settings = this_report_settings) #this


      #pull needed and reformat
      fcs_list[[i]] <- reportdata$modeling_results_data %>%
        #get forecasts only
        dplyr::filter(.data$series == "fc") %>%
        #get base date of report ('current date' in relation to forecast)
        dplyr::mutate(preadj_date = this_dt,
                      #how many weeks ahead is the prediction (not adjusting for reporting lag yet)
                      timestep_ahead_orig = difftime(.data$obs_date, .data$preadj_date) %>%
                                          as.numeric(units = "weeks")) %>%
        #don't need 0 week predictions (same week)
        dplyr::filter(.data$timestep_ahead_orig > 0)

      #force garbage collection
      rm(reportdata)
      gc()

    } #end timestep loop

    #have list of dataframes
    #collapse/bindrows
    fcs_only <- dplyr::bind_rows(fcs_list) %>%
      #nicely arrange
      dplyr::arrange(!!quo_groupfield, .data$timestep_ahead_orig, .data$obs_date)


    #join with the obs only extract to get observation series
    fc_join <- fcs_only %>%
      dplyr::left_join(obs_only,
                       #NSE fun
                       by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                               "obs_date"),
                                             c(rlang::as_name(quo_groupfield),
                                               "obs_date")))

    #make all the reporting_lag adjustments
    # basically, we ran extra forecast future steps
    # so we now can simply shift everything backwards except for averageweek
    if (this_model == "naive-averageweek"){
      fc_join <- fc_join %>%
        dplyr::mutate(run_date = .data$preadj_date,
                      #timestep_ahead is meaningless for average week.
                      # NA may cause unexpected results with grouping, so replace with 0
                      timestep_ahead = 0,
                      #Add column for showing reporting_lag
                      reporting_lag = reporting_lag)
    } else {
      fc_join <- fc_join %>%
        dplyr::mutate(run_date = .data$preadj_date - lubridate::weeks(reporting_lag),
                      timestep_ahead = .data$timestep_ahead_orig - reporting_lag,
                      #Add column for showing reporting_lag
                      reporting_lag = reporting_lag) %>%
        #filter out the timesteps that are now less than 1 step
        dplyr::filter(.data$timestep_ahead > 0)
    }


    #Filter to report weeks (trim off edges gathered b/c of weeks_ahead, etc.)
    fc_trim <- fc_join %>%
      dplyr::filter(dplyr::between(.data$obs_date,
                            date_start,
                            date_start + lubridate::weeks(total_timesteps-1)))


    ## Calculate statistics
    val_results <- calc_val_stats(fc_trim,
                                  quo_groupfield,
                                  per_timesteps,
                                  dots)

    #add results to list by name
    all_validations[[this_model]] <- val_results

  } #end model loop



  #Get skill test list of results
  if (skill_test == TRUE){
    #calc skill comparison statistics
    skill_overall <- calc_skill(get_overall_validations(all_validations))
    skill_grouping <- calc_skill(get_group_validations(all_validations), quo_groupfield)
    skill_scores <- create_named_list(skill_overall, skill_grouping)

    val_return <- create_named_list(skill_scores,
                                    validations = all_validations,
                                    metadata)
  } else {
    #just the one model validation datasets
    val_return <- create_named_list(all_validations,
                                    metadata)
  }

  message("Validation run finished.")
  val_return

} #end run validation



#'Calculate validation statistics from forecast results.
#'
#'Helper function to calculate the validation statistics from each model run.
#'Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Proportion of
#'observations in in prediction interval, and R^2. Calculates it both at a
#'global model level per timestep ahead, and at a geographical grouping level
#'per timestep ahead. Also calculates a timeseries of evaluation metrics at
#'every per_timesteps number of timesteps per grouping (if applicable) and
#'timestep_ahead.
#'
#'@param fc_trim The forecast results of one model type, combined with observed
#'  values, trimmed to user requested date range.
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_validation()/run_epidemia().
#'@param per_timesteps When creating a timeseries of validation results, create
#'  a moving window with per_timesteps width number of time points. Should be a
#'  minimum of 10 timesteps.
#'@param dots The non-required arguments to run_validation() for developer
#'  testing.
#'
#'@return A named list of validation statistic results: validation_overall,
#'  validation_grouping, validation_timeseries, validation_perweek
#'
calc_val_stats <- function(fc_trim, quo_groupfield, per_timesteps, dots){
  # MAE: mean(|obs - pred|)
  # RMSE: sqrt(mean((obs - pred)^2))
  # R2 (R^2): 1 - SSE/TSS.  SSE = sum((obs-pred)^2). TSS = sum((obs - mean(obs))^2).
  #     B/c involves mean of group of observations, must be calculated after grouping

  #Removed
  # Proportion in Interval: 1/T if inside, summed. Over all non-NA entries.

  #per line stats
  fc_stats <- fc_trim %>%
    dplyr::mutate(diff = .data$obs - .data$value,
                  absdiff = abs(.data$diff),
                  diffsq = .data$diff ^ 2)
                  #,predinterval = ifelse(obs >= lower & obs <= upper, TRUE, FALSE))


  #overall timestep_ahead
  validation_overall <- fc_stats %>%
    dplyr::group_by(.data$timestep_ahead) %>%
    #Now calc TSS part of R2
    dplyr::mutate(meanobs = mean(.data$obs, na.rm = TRUE),
                  total_squares = (.data$obs - .data$meanobs)^2) %>%
    #stat calc
    dplyr::summarize(MAE = mean(.data$absdiff, na.rm = TRUE),
                     MSE = mean(.data$diffsq, na.rm = TRUE),
                     #prop_interval = sum(predinterval, na.rm = TRUE) / sum(!is.na(predinterval)),
                     SSE = sum(.data$diffsq, na.rm = TRUE),
                     TSS = sum(.data$total_squares, na.rm = TRUE)) %>%
    #and mutate for final calc
    dplyr::mutate(RMSE = sqrt(.data$MSE),
                  R2 = 1 - (.data$SSE / .data$TSS)) %>%
    #drop unneeded columns
    dplyr::select(-.data$SSE, -.data$TSS, -.data$MSE)



  #overall timestep_ahead by grouping
  validation_grouping <- fc_stats %>%
    dplyr::group_by(!!quo_groupfield, .data$timestep_ahead) %>%
    #Now calc TSS part of R2
    dplyr::mutate(meanobs = mean(.data$obs, na.rm = TRUE),
                  total_squares = (.data$obs - .data$meanobs)^2) %>%
    #stat calc
    dplyr::summarize(MAE = mean(.data$absdiff, na.rm = TRUE),
                     MSE = mean(.data$diffsq, na.rm = TRUE),
                     #prop_interval = sum(predinterval, na.rm = TRUE) / sum(!is.na(predinterval)),
                     SSE = sum(.data$diffsq, na.rm = TRUE),
                     TSS = sum(.data$total_squares, na.rm = TRUE)) %>%
    #and mutate for final calc
    dplyr::mutate(RMSE = sqrt(.data$MSE),
                  R2 = 1 - (.data$SSE / .data$TSS)) %>%
    #drop unneeded columns
    dplyr::select(-.data$SSE, -.data$TSS, -.data$MSE)



  #timeseries calculations
  # minimum of ~10 timesteps per summary
  # ROLLING window
  validation_timeseries <- fc_stats %>%
    dplyr::group_by(!!quo_groupfield, .data$timestep_ahead) %>%
    #rollapply for get mean of obs
    dplyr::mutate(meanobs = zoo::rollmeanr(x = .data$obs,
                                           k = per_timesteps,
                                           fill = NA),
                  total_squares = (.data$obs - .data$meanobs)^2,
                  MAE = zoo::rollmeanr(x = .data$absdiff,
                                       k = per_timesteps,
                                       fill = NA),
                  MSE = zoo::rollmeanr(x = .data$diffsq,
                                       k = per_timesteps,
                                       fill = NA),
                  RMSE = sqrt(.data$MSE),
                  #prop_interval = zoo::rollsumr(x = predinterval,
                  #                              k = per_timesteps,
                  #                              fill = NA) /
                  #  zoo::rollsumr(x = !is.na(predinterval),
                  #                k = per_timesteps,
                  #                fill = NA),
                  SSE = zoo::rollsumr(x = .data$diffsq,
                                      k = per_timesteps,
                                      fill = NA),
                  TSS = zoo::rollsumr(x = .data$total_squares,
                                      k = per_timesteps,
                                      fill = NA),
                  R2 = 1 - (.data$SSE / .data$TSS)) %>%
    #rename columns to be clearer
    dplyr::rename(forecast = .data$value,
                  observed = .data$obs) %>%
    # drop unneeded columns
    dplyr::select(-.data$series, -.data$preadj_date, -.data$timestep_ahead_orig, -.data$run_date,
                  -.data$diff, -.data$absdiff, -.data$diffsq, -.data$meanobs,
                  -.data$total_squares, -.data$MSE, -.data$SSE, -.data$TSS) %>%
    # for now, drop R2 until can figure out how to include better
    dplyr::select(-.data$R2)



  #return all
  val_stats <- create_named_list(validation_overall,
                                 validation_grouping,
                                 validation_timeseries,
                                 validation_perweek = fc_stats)

  # # and raw data with hidden option
  # #possibly make "time series" version for clean full data table
  # if (!is.null(dots[['raw_data']])){
  #   if (dots[['raw_data']] == TRUE){
  #     val_stats <- create_named_list(validation_overall,
  #                                    validation_grouping,
  #                                    validation_timeseries,
  #                                    raw_stats = fc_stats)
  #   } #end raw data TRUE
  # } else {
  #   #normal return with just results
  #   val_stats <- create_named_list(validation_overall,
  #                                  validation_grouping,
  #                                  validation_timeseries)
  # }

} #end calc_val_stats()


#' Get overall model validation statistics
#'
#' Small function to pull out just overall validation statistics.
#'
#' @param validations The set of validation statistics produced by
#'   run_validation() - only the list of validation data sets, not including the skill metrics.
#'
#' @return A list of tibbles containing only the model overall statistics (and
#'   not including the geographic grouping results, if present).
#'
#' @export
#'
get_overall_validations <- function(validations){
  lapply(validations, `[[`, "validation_overall")
}


#' Get geographic grouping model validation statistics
#'
#' Small function to pull out just the geographic grouping validation statistics.
#'
#' @param validations The set of validation statistics produced by
#'   run_validation() - only the list of validation data sets, not including the skill metrics.
#'
#' @return A list of tibbles containing only the model geographic grouping statistics.
#'
#' @export
#'
get_group_validations <- function(validations){
  lapply(validations, `[[`, "validation_grouping")
}



#' Calculate model skill comparison statistics
#'
#' Helper function to calculate the relative improvement of the forecast over the specified naive model.
#' Skill score = (score_fc - score_naive) / (score_perfect - score_naive)
#' Skill metric has an upper bound of 1. No improvement is 0. Lower bound depends on statistic.
#'
#'@param fc_stat The forecast model statistic value.
#'@param naive_stat The naive model statistic value (same statistic as forecast model).
#'@param perfect_stat The value of a perfect score for that stastistic.
#'
#'@return Skill score: the relative improvement the forecast model has over the naive model.
#'
#'@export
#'
calc_skill_stat <- function(fc_stat, naive_stat, perfect_stat){
  skill_stat <- (fc_stat - naive_stat) / (perfect_stat - naive_stat)
}


#' Calculate the forecast model skill score compared to the naive model predictions.
#'
#'@param val_list A list of 3 datasets of validation results: the first is the forecast model, the following two are the naive model results, as created by binding the results of calc_val_stats() in run_validation().
#'@param grp Optional inclusion of quo_groupfield when calculating skill scores by groupfield.
#'
#'@return Single dataset with skill scores of the main forecast model against each of the naive models, per timestep ahead, and optionally, per geographic grouping
#'
calc_skill <- function(val_list, grp = NULL){

  #separate out, rename columns, and join/crossing
  val_fc <- val_list[[1]] %>%
    dplyr::rename(fc_MAE = .data$MAE,
                  fc_RMSE = .data$RMSE,
                  #fc_prop_interval = prop_interval,
                  fc_R2 = .data$R2) %>%
    dplyr::select(dplyr::group_cols(), .data$timestep_ahead, dplyr::starts_with("fc_"))

  val_np <- val_list$`naive-persistence` %>%
    dplyr::rename(np_MAE = .data$MAE,
                  np_RMSE = .data$RMSE,
                  #np_prop_interval = prop_interval,
                  np_R2 = .data$R2) %>%
    dplyr::select(dplyr::group_cols(), .data$timestep_ahead, dplyr::starts_with("np_"))

  val_naw <- val_list$`naive-averageweek` %>%
    dplyr::rename(naw_MAE = .data$MAE,
           naw_RMSE = .data$RMSE,
           #naw_prop_interval = prop_interval,
           naw_R2 = .data$R2) %>%
    #no timestep_ahead for average week, all same
    dplyr::select(dplyr::group_cols(), dplyr::starts_with("naw_"))

  #appropriate joins
  if (is.null(grp)){
    #join together
    val_join <- val_fc %>%
      #join with persistence
      dplyr::left_join(val_np,
                       by = "timestep_ahead") %>%
      #join with average week (1 value to all timesteps ahead)
      tidyr::crossing(val_naw)
  } else {
    #else join with groupfield
    #join together
    val_join <- val_fc %>%
      #join with persistence
      dplyr::left_join(val_np,
                       #NSE fun
                       by = rlang::set_names(c(rlang::as_name(grp),
                                               "timestep_ahead"),
                                             c(rlang::as_name(grp),
                                               "timestep_ahead"))) %>%
      #join with average week (1 value to all timesteps ahead)
      dplyr::left_join(val_naw,
                       by = rlang::set_names(rlang::as_name(grp),
                                             rlang::as_name(grp)))
  } #end joinings

  #perfect skill metrics
  perfect_MAE <- 0
  perfect_RMSE <- 0
  #perfect_prop_interval <- 1
  perfect_R2 <- 1

  #calc skill metrics of fc model to each of naive models
  val_skill <- val_join %>%
    dplyr::mutate(skill_MAE_persistence = calc_skill_stat(.data$fc_MAE, .data$np_MAE, perfect_MAE),
           skill_RMSE_persistence = calc_skill_stat(.data$fc_RMSE, .data$np_RMSE, perfect_RMSE),
           #skill_interval_persistence = calc_skill_stat(.data$fc_prop_interval, .data$np_prop_interval,
           #                                             perfect_prop_interval),
           skill_R2_persistence =  calc_skill_stat(.data$fc_R2, .data$np_R2, perfect_R2),
           skill_MAE_averageweek = calc_skill_stat(.data$fc_MAE, .data$naw_MAE, perfect_MAE),
           skill_RMSE_averageweek = calc_skill_stat(.data$fc_RMSE, .data$naw_RMSE, perfect_RMSE),
           #skill_interval_averageweek = calc_skill_stat(.data$fc_prop_interval, .data$naw_prop_interval,
           #                                             perfect_prop_interval),
           skill_R2_averageweek =  calc_skill_stat(.data$fc_R2, .data$naw_R2, perfect_R2)) %>%
    #select final stats only
    dplyr::select(dplyr::group_cols(), .data$timestep_ahead, dplyr::starts_with("skill_"))

  val_skill
}



#' Save overall model validation statistics
#'
#' Small function to pull out just overall validation statistics and save to
#' csv.
#'
#' @param validations The set of validation statistics produced by
#'   run_validation() - only the list of validation data sets, not including the skill metrics.
#' @param save_file File name to save results into csv format
#'
#' @return A csv file containing only the model overall statistics (and not
#'   including the geographic grouping results, if present).
#'
#' @export
#'
save_overall_validations <- function(validations, save_file){
  lapply(validations, `[[`, "validation_overall") %>%
    dplyr::bind_rows(.id = "model") %>%
    readr::write_csv(save_file)
}


#' Save geographic grouping model validation statistics
#'
#' Small function to pull out validation statistics per geographic grouping and
#' save to csv.
#'
#' @param validations The set of validation statistics produced by
#'   run_validation() - only the list of validation data sets, not including the skill metrics.
#' @param save_file File name to save results into csv format
#'
#' @return A csv file containing the model validation statistics for the
#'   geographic grouping results.
#'
#' @export
#'
save_geog_validations <- function(validations, save_file){
  lapply(validations, `[[`, "validation_grouping") %>%
    dplyr::bind_rows(.id = "model") %>%
    readr::write_csv(save_file)
}
