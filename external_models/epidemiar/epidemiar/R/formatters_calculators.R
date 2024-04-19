#Formatting and Report section data calculators

## Environmental Data for report
#' Formats environmental data for report timeseries.
#'
#'@param env_ext_data An environmental dataset extended into the
#'  future/forecast period with estimated values for the environmental
#'  variables. The env_data_extd object returned by run_forecast().
#'@param env_ref_data Historical averages by week of year for environmental
#'  variables. Used in extended environmental data into the future for long
#'  forecast time, to calculate anomalies in early detection period, and to
#'  display on timeseries in reports.
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_epidemia().
#'@param quo_obsfield Quosure of user given field name of the environmental data
#'  variables
#'@param env_used List of environmental variables that were used in
#'  the modeling.
#'@param env_info Lookup table for environmental data - reference creation
#'  method (e.g. sum or mean), report labels, etc.
#'@param epi_date_type String indicating the standard (WHO ISO-8601 or CDC epi
#'  weeks) that the weeks of the year in epidemiological and environmental
#'  reference data use ["ISO" or "CDC"]. <<>>
#'@param report_dates Internally generated set of report date information: min,
#'  max, list of dates for full report, known epidemiological data period,
#'  forecast period, and early detection period.
#'
#'@return Data set of multiple timeseries for the used environmental variables
#'  during the report period for each geographic unit. Returned as
#'  environ_timeseries in the run_epidemia() output.
#'
environ_report_format <- function(env_ext_data,
                                  env_ref_data,
                                  quo_groupfield,
                                  quo_obsfield,
                                  env_used,
                                  env_info,
                                  epi_date_type,
                                  report_dates){

  #for adding week, year fields
  week_type <- dplyr::case_when(
    epi_date_type == "weekISO" ~ "ISO",
    epi_date_type == "weekCDC"  ~ "CDC",
    #default NA
    TRUE             ~ NA_character_)

  #daily env data
  env_data_varused <- env_ext_data %>%
    dplyr::filter(!!quo_obsfield %in% env_used)

  #reference/climatology environmental data
  env_ref_varused <- env_ref_data %>%
    dplyr::filter(!!quo_obsfield %in% env_used)

  ##properly summarize to weekly (from daily)
  env_data_varused_sum <- env_data_varused %>%
    #get reference/summarizing method from user supplied env_info
    dplyr::left_join(env_info %>%
                       dplyr::select(!!quo_obsfield, .data$reference_method),
                     by = rlang::set_names(rlang::as_name(quo_obsfield),
                                           rlang::as_name(quo_obsfield))) %>%
    #add date fields
    epidemiar::add_datefields(week_type) %>%
    #trim dates to reduce processing (dates are rough, technically just need week prior to start. 8 is not magical)
    dplyr::filter(.data$obs_date >= report_dates$full$min - 8 & .data$obs_date <= report_dates$full$max + 8) %>%
    #group by grouping, env var, and date week
    dplyr::group_by(!!quo_groupfield, !!quo_obsfield, .data$year_epidemiar, .data$week_epidemiar) %>%
    #calculate with case_when at row level (fx is not vectorized, so can't be used inside summarize)
    dplyr::mutate(val_epidemiar = dplyr::case_when(
      .data$reference_method == "sum"  ~ sum(.data$val_epidemiar, na.rm = TRUE),
      .data$reference_method == "mean" ~ mean(.data$val_epidemiar, na.rm = TRUE),
      #default is mean
      TRUE                       ~ mean(.data$val_epidemiar, na.rm = TRUE))) %>%
    #now summarize
    #max Date of that week is how the weekly dates are set up
    dplyr::summarize(obs_date = max(.data$obs_date),
                     #val_epi is the same for the whole grouped set, so just taking the first value
                     val_epidemiar = dplyr::first(.data$val_epidemiar),
                     #will be same throughout week
                     reference_method = dplyr::first(.data$reference_method),
                     #observed/interpolated/extended -- Mode, whatever source was most often that week.
                     data_source = Mode(.data$data_source, na.rm = TRUE)) %>%
    #ungroup to end
    dplyr::ungroup()

  #filter exact dates
  environ_timeseries <- env_data_varused_sum %>%
    dplyr::filter(.data$obs_date >= report_dates$full$min & .data$obs_date <= report_dates$full$max) %>%
    dplyr::arrange(!!quo_groupfield, .data$obs_date, !!quo_obsfield)

  # add climatology data
  # climatology is based on week number
  #   (should have been set up with the same week type as was selected when ref data was created)
  environ_timeseries <- environ_timeseries %>%
    #join
    dplyr::left_join(env_ref_varused %>%
                       dplyr::select(!!quo_obsfield, !!quo_groupfield,
                                     .data$week_epidemiar,
                                     .data$ref_value, dplyr::starts_with("ref_")),
                     #NSE fun
                     by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                             rlang::as_name(quo_obsfield),
                                             "week_epidemiar"),
                                           c(rlang::as_name(quo_groupfield),
                                             rlang::as_name(quo_obsfield),
                                             "week_epidemiar")))
}


## Setting up summary data
#' Creates early detection and early warning alerts levels for each geographic group.
#'
#'@param ed_res Event detection results from run_event_detection().
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_epidemia().
#'@param report_dates Internally generated set of report date information: min,
#'  max, list of dates for full report, known epidemiological data period,
#'  forecast period, and early detection period.
#'
#'@return Data set of early detection and early warning alert summaries for each
#'  geographic group. Returned as summary_data in the run_epidemia() output.
#'
create_summary_data <- function(ed_res,
                                quo_groupfield,
                                report_dates){

  #levels
  alert_level <- c("Low", "Medium", "High")

  #if early detection period was defined (ed_summary_period > 0)
  if (!is.na(report_dates$ed_sum$min)) {
    #Early Detection
    ed_summary <- ed_res %>%
      #get the alert series for all early detection
      dplyr::filter(.data$series == "ed") %>%
      #filter to defined early detection period
      dplyr::filter(.data$obs_date %in% report_dates$ed_sum$seq) %>%
      #group (because need to look at period per group level)
      dplyr::group_by(!!quo_groupfield) %>%
      #summarize to 1 obs per grouping
      dplyr::summarize(ed_alert_count = dplyr::if_else(all(is.na(.data$value)), NA_real_, sum(.data$value, na.rm = TRUE))) %>%
      # create 3 levels (0, 1, 2 = >1)
      dplyr::mutate(warning_level = dplyr::if_else(.data$ed_alert_count > 1, 2, .data$ed_alert_count),
                    #factor to label
                    ed_sum_level = factor(.data$warning_level, levels = 0:2,
                                          labels = alert_level, ordered = TRUE)) %>%
      #ungroup
      dplyr::ungroup() %>%
      #select minimal cols
      dplyr::select(!!quo_groupfield, .data$ed_alert_count, .data$ed_sum_level)

  } else {
    #create NA ED results for when ed_summary_period = 0
    ed_summary <- ed_res %>%
      #create an entry for each geogroup, for creating NA results)
      dplyr::select(!!quo_groupfield) %>%
      dplyr::group_by(!!quo_groupfield) %>%
      unique() %>%
      #add in NA results
      dplyr::mutate(ed_alert_count = NA,
                    ed_sum_level = NA) %>%
      #confirm same output structure
      #ungroup
      dplyr::ungroup() %>%
      #select minimal cols
      dplyr::select(!!quo_groupfield, .data$ed_alert_count, .data$ed_sum_level)
  }


  #Early Warning: ED results on forecast
  ew_summary <- ed_res %>%
    #get the alert series
    dplyr::filter(.data$series == "ew",
                  #get the forecast results ##not needed anymore b/c of new ew series, but just for completeness
                  .data$obs_date %in% report_dates$forecast$seq) %>%
    #group
    dplyr::group_by(!!quo_groupfield) %>%
    #summarize to 1 obs per grouping
    dplyr::summarize(ew_alert_count = dplyr::if_else(all(is.na(.data$value)), NA_real_, sum(.data$value, na.rm = TRUE))) %>%
    # create 3 levels (0, 1, 2 = >1)
    dplyr::mutate(warning_level = dplyr::if_else(.data$ew_alert_count > 1, 2, .data$ew_alert_count),
                  #factor to label
                  ew_level = factor(.data$warning_level, levels = 0:2,
                                    labels = alert_level, ordered = TRUE)) %>%
    #ungroup
    dplyr::ungroup() %>%
    #select minimal cols
    dplyr::select(!!quo_groupfield, .data$ew_alert_count, .data$ew_level)

  #join results
  summary_data <- dplyr::inner_join(ed_summary, ew_summary,
                                    by = rlang::set_names(rlang::as_name(quo_groupfield),
                                                          rlang::as_name(quo_groupfield)))

  summary_data
}

#' Creates summary of disease incidence in early detection period.
#'
#'@param obs_res Formatted dataset of observed disease incidence.
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_epidemia().
#'@param report_dates Internally generated set of report date information: min,
#'  max, list of dates for full report, known epidemiological data period,
#'  forecast period, and early detection period.
#'
#'@return Mean disease incidence per geographic group during the early detection
#'  period, returned as epi_summary in the run_epidemia() ouput.
#'
create_epi_summary <- function(obs_res,
                               quo_groupfield,
                               report_dates){

  #if early detection period was defined (ed_summary_period > 0)
  if (!is.na(report_dates$ed_sum$min)) {
    epi <- obs_res %>%
      #epi data is weekly, get the data for the early detection summary period
      dplyr::filter(.data$obs_date %in% report_dates$ed_sum$seq) %>%
      #group by groupings
      dplyr::group_by(!!quo_groupfield) %>%
      #get mean incidence/cases (which ever user had selected will be in value field)
      dplyr::summarize(mean_epi = mean(.data$value, na.rm = TRUE))
  } else {
    #create NA epi results for when ed_summary_period = 0
    epi <- obs_res %>%
      #create an entry for each geogroup, for creating NA results)
      dplyr::select(!!quo_groupfield) %>%
      dplyr::group_by(!!quo_groupfield) %>%
      unique() %>%
      #add NA result
      dplyr::mutate(mean_epi = NA)
  }

  epi

}


## Calculate anomalies
#' Calculates and summarizes environmental anomalies in the early detection period.
#'
#'@param env_ts Environmental timeseries dataset as output from environ_report_format().
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_epidemia().
#'@param quo_obsfield Quosure of user given field name of the environmental data
#'  variables.
#'@param report_dates Internally generated set of report date information: min,
#'  max, list of dates for full report, known epidemiological data period,
#'  forecast period, and early detection period.
#'
#' @return These data are the recent (during the early detection period)
#'   differences (anomalies) of the environmental variable values from the
#'   climatology/reference mean. Note: these are not the same as daily
#'   environmental anomalies calculated as residuals from GAM in anomalize_env()
#'   as part of forecasting.
#'
calc_env_anomalies <- function(env_ts,
                               quo_groupfield,
                               quo_obsfield,
                               report_dates){

  #if early detection period was defined (ed_summary_period > 0)
  if (!is.na(report_dates$ed_sum$min)) {
    #environmental observed data in early detection period
    env_ed <- env_ts %>%
      # only mapping those in the early detection period
      dplyr::filter(.data$obs_date %in% report_dates$ed_sum$seq) %>%
      # do not use "Extended" or "Interpolated" data, only "Observed"
      dplyr::mutate(val_epidemiar = dplyr::if_else(.data$data_source == "Observed", .data$val_epidemiar, NA_real_))

    # anomalies
    anom_env <- env_ed %>%
      #group
      dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
      # anomaly value is observed value minus the ref value from env_ref
      dplyr::mutate(anom = .data$val_epidemiar - .data$ref_value) %>%
      # summarized over ED period
      dplyr::summarize(anom_ed_mean = mean(.data$anom, na.rm = TRUE)) %>%
      dplyr::ungroup()

  } else {
    #create NA results for when there is no early detection period
    anom_env <- env_ts %>%
      #create an entry for each geogroup, for creating NA results)
      dplyr::select(!!quo_groupfield, !!quo_obsfield) %>%
      dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
      unique() %>%
      #add NA result
      dplyr::mutate(anom_ed_mean = NA) %>%
      dplyr::ungroup()

  }

  anom_env
}


#' Formats report_settings for including in metadata part of final report data
#'
#'@param rpt_settings Report settings after processing defaults and matching.
#'
#'@return Named list of report_settings, in alphabetical order and no developer settings
#'
format_report_settings <- function(rpt_settings){
  #order alphabetically
  clean_settings <- rpt_settings[order(names(rpt_settings))]

  #remove dev IF no dev settings were changed from default
  #so if dev settings all default, then remove
  if (rpt_settings[["dev_fc_fit_freq"]] == "once" &
      is.null(rpt_settings[["dev_fc_formula"]])){
    clean_settings <- clean_settings[!grepl("^dev", names(clean_settings))]
  }

  clean_settings
}
