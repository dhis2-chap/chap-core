# helper/subfunctions related to forecasting

#' Pull only model env variables.
#'
#'@param env_var Extract from report_settings$env_var, a user list of environmental variables to attempt to use for modeling
#'
#'@inheritParams run_epidemia
#'@inheritParams run_forecast
#'
#'@return List of environmental variables that were used in the
#'  modeling (had to be both listed in model variables input file and present the
#'  env_data dataset).
#'
pull_model_envvars <- function(env_data,
                               quo_obsfield,
                               env_var){

  #pull variables into list
  model_vars <- env_var %>% dplyr::pull(!!quo_obsfield)

  #filter env_data for those model_vars
  env_data <- env_data %>%
    dplyr::filter(!!quo_obsfield %in% model_vars)
}

#' Extend environmental data into the future.
#'
#'@param epi_date_type Extract from `report_settings$epi_date_type`
#'@param env_variables_used List of environmental variables that were used in
#'  the modeling (in `report_settings$env_var` & found in env_data and env_info)
#'
#'@inheritParams run_epidemia
#'@inheritParams run_forecast
#'
#'@return Environmental dataset, with data extended into the future forecast
#'  period. Unknown environmental data with runs of < 2 weeks is
#'  filled in with last known data (i.e. "persistence" method, using the mean of
#'  the previous week of known data). For missing data runs more than 2 weeks, the
#'  values are filled in using a progressive blend of the the mean of the last
#'  known week and the historical means.
#'
extend_env_future <- function(env_data,
                              quo_groupfield,
                              quo_obsfield,
                              quo_valuefield,
                              env_ref_data,
                              env_info,
                              fc_model_family,
                              #pull from report_settings
                              epi_date_type,
                              #calculated/internal
                              valid_run,
                              groupings,
                              env_variables_used,
                              report_dates){

  # Extend environmental data into the future forecast time period, while
  # also dealing with any other missing environmental data.

  # Progressive blend (over # of missing weeks) of
  # mean of last known week & historical/climatological means (weekly value)
  # but only if missing run is larger than 2 weeks * 7 = 14 days
  # if less than, just use persistence/carry forward/last known value
  # E.g. if 20 missing in a run:
  # 1 was filled in with previous week mean (recent value) for 'mean' type, 14 days for 'sum' type
  # 2: 19/20 recent + 1/20 historical, 3: 18/20 recent + 2/20 historical, ... 20: 1/20 recent + 19/20 historical.
  # Will ALWAYS include part of recent known data (relevant if recent patterns are departure from climate averages)

  # switch epi_date_type to week_type needed for add_datefields()
  week_type <- dplyr::case_when(
    epi_date_type == "weekISO" ~ "ISO",
    epi_date_type == "weekCDC"  ~ "CDC",
    #default as if mean
    TRUE             ~ NA_character_)

  #Do not need data past end of forecast period (if exists)
  env_trim <- env_data %>%
    dplyr::filter(.data$obs_date <= report_dates$forecast$max)


  #Possible situations:
  #Missing data in 'past/known' period, or future unknown data,
  # or both, or neither

  #Calculate full/complete data table
  #combination of all groups, env vars, and dates (DAILY)
  env_complete <- tidyr::crossing(obs_date = seq.Date(from = min(env_trim$obs_date),
                                                      to = report_dates$forecast$max,
                                                      by = "day"),
                                  group_temp = groupings,
                                  obs_temp = env_variables_used)
  #and fix names with NSE
  env_complete <- env_complete %>%
    dplyr::rename(!!rlang::as_name(quo_groupfield) := .data$group_temp,
                  !!rlang::as_name(quo_obsfield) := .data$obs_temp)

  #could have ragged env data per variable per grouping
  #so, antijoin with env_known_fill first to get the actually missing rows
  env_missing <- env_complete %>%
    dplyr::anti_join(env_trim, by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                                       rlang::as_name(quo_obsfield),
                                                       "obs_date"),
                                                     c(rlang::as_name(quo_groupfield),
                                                       rlang::as_name(quo_obsfield),
                                                       "obs_date")))


  if (nrow(env_missing) > 1 | any(is.na(env_trim$val_epidemiar))){
    #some amount of missing data
    #first test is implicit (missing row), second is explicit (row exists, but value NA)

    #bind with existing data (NAs for everything else)
    # (env_future name ~ env plus future period, hold over from when this only did future portion)
    env_future <- dplyr::bind_rows(env_trim, env_missing) %>%
      #mark which are about to be filled in
      dplyr::mutate(data_source = ifelse(is.na(.data$val_epidemiar), "Imputed", "Observed"))

    #Optimizing for speed for validation runs with naive models, skip unneeded

    if (valid_run == TRUE &
        (fc_model_family == "naive-persistence" | fc_model_family == "naive-averageweek")){

      #Missing environmental data is fine for naive models
      # as they do not use that data
      # and validation runs do not return env data
      env_extended_final <- env_future

    } else {
      #need to fill in missing data

      #function for rle needed to honor group_bys
      # computes an rle based on if value is NA or not
      # returns the number of rows in the run
      get_rle_na_info <- function(x){
        x_na_rle <- rle(is.na(x))
        run_id <- rep(seq_along(x_na_rle$lengths), times = x_na_rle$lengths)
        run_tot <- rep(x_na_rle$lengths, times = x_na_rle$lengths)
        dplyr::as_tibble(create_named_list(run_id, run_tot))
      }


      env_na_rle <- env_future %>%
        dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
        #make doubly sure in sorted date order
        dplyr::arrange(.data$obs_date) %>%
        #since adding multiple columns, use do instead of mutate
        dplyr::do(cbind(., get_rle_na_info(.$val_epidemiar))) %>%
        #add a groupby with the new run ID
        dplyr::group_by(!!quo_groupfield, !!quo_obsfield, .data$run_id) %>%
        #creates an index of where that row is in the run
        dplyr::mutate(id_in_run = seq_along(.data$val_epidemiar)) %>%
        #ungroup to end set
        dplyr::ungroup()

      ##Get env info and ref data
      #Prep ref data - get only used vars
      env_ref_varused <- env_ref_data %>%
        dplyr::filter(!!quo_obsfield %in% env_variables_used)

      #joins for ref summary type, and summary for week
      env_join_ref <- env_na_rle %>%
        #add week, year fields
        epidemiar::add_datefields(week_type) %>%
        #get reference/summarizing method from user supplied env_info
        dplyr::left_join(env_info %>%
                           dplyr::select(!!quo_obsfield, .data$reference_method),
                         by = rlang::set_names(rlang::as_name(quo_obsfield),
                                               rlang::as_name(quo_obsfield))) %>%
        #get weekly ref value
        dplyr::left_join(env_ref_varused %>%
                           dplyr::select(!!quo_obsfield, !!quo_groupfield,
                                         .data$week_epidemiar, .data$ref_value),
                         #NSE fun
                         by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                                 rlang::as_name(quo_obsfield),
                                                 "week_epidemiar"),
                                               c(rlang::as_name(quo_groupfield),
                                                 rlang::as_name(quo_obsfield),
                                                 "week_epidemiar")))


      #find 1st NA, then take mean of previous week, input for that day
      #first NA now can be found with is.na(val_epidemiar) & id_in_run == 1
      #use zoo::rollapply for mean
      # for 'mean' type, last 7 days
      # for 'sum' type (e.g. highly variable precip), last 14 days

      #Fill in first NA of a run with the mean of previous
      env_na1fill <- env_join_ref %>%
        #confirm proper grouping
        dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
        # confirm proper sorting
        dplyr::arrange(!!quo_groupfield, !!quo_obsfield, .data$obs_date) %>%
        #create a 1 day lag variable since need previous days not including current
        dplyr::mutate(val_lag1 = dplyr::lag(.data$val_epidemiar, n = 1),
                      #zoo:rollapply to calculate mean of last 7 days (week) on lagged var
                      mean_for_mean_type_lag1 = zoo::rollapply(data = .data$val_lag1,
                                                  width = 7,
                                                  FUN = mean,
                                                  align = "right",
                                                  na.rm = TRUE,
                                                  #fill important to align properly with mutate
                                                  fill = NA),
                      mean_for_sum_type_lag1 = zoo::rollapply(data = .data$val_lag1,
                                                  width = 14,
                                                  FUN = mean,
                                                  align = "right",
                                                  na.rm = TRUE,
                                                  #fill important to align properly with mutate
                                                  fill = NA),
                      #ifelse to find the first NA
                      val_epidemiar = ifelse(is.na(.data$val_epidemiar) & .data$id_in_run == 1,
                                             dplyr::case_when(
                                               #for mean type, for sum type
                                               .data$reference_method == "mean" ~ .data$mean_for_mean_type_lag1,
                                               .data$reference_method == "sum" ~ .data$mean_for_sum_type_lag1,
                                              #default (nothing currently using)
                                                TRUE ~ .data$mean_for_mean_type_lag1),
                                             #if not first NA, then use original val_epidemiar value
                                             .data$val_epidemiar)) %>%
        #drop unneeded lag column
        dplyr::select(-c(.data$val_lag1, .data$mean_for_mean_type_lag1, .data$mean_for_sum_type_lag1))


      #calculate NA missing values using carry|blend
      env_filled <- env_na1fill %>%
        #order very important for filling next step
        dplyr::arrange(!!quo_groupfield, !!quo_obsfield, .data$obs_date) %>%
        dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
        #propagate last known value down rows
        dplyr::mutate(last_known = .data$val_epidemiar) %>%
        #fill down, so missing weeks has "last known value" IN row for calculations
        tidyr::fill(.data$last_known, .direction = "down") %>%
        #calculate parts (for all, will only use when needed)
        # with progressive blending based on id_in_run and run_tot
        dplyr::mutate(recent_modifier = (.data$run_tot - .data$id_in_run - 1) / .data$run_tot,
                      recent_part = .data$recent_modifier * .data$last_known,
                      historical_modifier = (.data$id_in_run - 1) / .data$run_tot,
                      #historical is by week, so get pseudo-daily value depending on reference method,
                      # i.e. how to summarize a week of data
                      historical_value = dplyr::case_when(
                        .data$reference_method == "mean" ~ .data$ref_value,
                        .data$reference_method == "sum"  ~ .data$ref_value / 7,
                        #default as if mean
                        TRUE             ~ .data$ref_value),
                      historical_part = .data$historical_modifier * .data$historical_value,
                      #testing
                      val_orig = .data$val_epidemiar,
                      #only fill NA values
                      val_epidemiar = ifelse(is.na(.data$val_epidemiar),
                                             #persist if <15 days, blend if greater
                                             ifelse(.data$run_tot < 15,
                                                    .data$last_known,
                                                    .data$recent_part + .data$historical_part),
                                             #if notNA, then use existing val_epidemiar value
                                             .data$val_epidemiar))

      #clean up
      env_extended_final <- env_filled %>%
        #remove all added columns to match original format
        dplyr::select(-c(.data$run_id, .data$run_tot, .data$id_in_run,
                         .data$week_epidemiar, .data$year_epidemiar,
                         .data$last_known,
                         .data$reference_method, .data$ref_value,
                         .data$recent_modifier, .data$recent_part,
                         .data$historical_modifier, .data$historical_value, .data$historical_part,
                         .data$val_orig)) %>%
        #fill everything except original value field
        #for any other column that got vanished during crossing, etc.
        tidyr::fill(dplyr::everything(),
                    -!!quo_valuefield, -!!quo_groupfield, -!!quo_obsfield,
                    .direction = "down") %>%
        #ungroup to end
        dplyr::ungroup()

    } #end else on valid run & naive models

    } else { #else on if missing rows
      #no missing data, just use trimmed environmental data set as given
      env_extended_final <- env_trim %>%
        dplyr::mutate(data_source = "Observed")
    }

  #several paths to get to an env_extended_final
  return(env_extended_final)

} # end extend_env_future




#' Extend epidemiology dataframe into future.
#'
#'@inheritParams run_epidemia
#'@inheritParams run_forecast
#'
#'@return Epidemiological dataset extended past the known epi data time range
#'  and into the future/forecast period. Case numbers are filled in the NA (to
#'  be forecasted), and the population is estimated in a persistence method.
#'
extend_epi_future <- function(epi_data,
                              quo_popfield,
                              quo_groupfield,
                              #calculated/internal
                              groupings,
                              report_dates){

  #extended epi data into future dates
  #for use in modeling later (results will be put elsewhere), this is for env and lags and modeling dataset

  #get future/forecast dates
  epi_future <- tidyr::crossing(obs_date = report_dates$forecast$seq,
                                group_temp = groupings)
  #and fix names with NSE
  epi_future <- epi_future %>%
    dplyr::rename(!!rlang::as_name(quo_groupfield) := .data$group_temp)

  #with fc_start_date, there MAY be observed data in future/forecast period
  #so antijoin and bind actual needed rows to avoid duplication
  epi_future_missing <- epi_future %>%
    dplyr::anti_join(epi_data, by = rlang::set_names(c(rlang::as_name(quo_groupfield),
                                                       "obs_date"),
                                                     c(rlang::as_name(quo_groupfield),
                                                       "obs_date")))

  #bind with exisiting data (NAs for everything else in epi_future)
  extended_epi <- dplyr::bind_rows(epi_data, epi_future_missing) %>%
    dplyr::arrange(!!quo_groupfield, .data$obs_date)

  #fill population down (if pop field given)
  if (!is.null(quo_popfield)) {
    extended_epi <- extended_epi %>%
      #per geographic group
      dplyr::group_by(!!quo_groupfield) %>%
      #fill population down ('persistence' fill, last known value carried forward)
      tidyr::fill(!!quo_popfield, .direction = "down") %>%
      #ungroup to finish
      dplyr::ungroup()
  }

  extended_epi
}


#' Format env data for modeling
#'
#'@param env_data_extd An environmental dataset extended into the
#'  future/forecast period with estimated values for the environmental
#'  variables.
#'
#'@inheritParams run_forecast
#'
#'@return An environmental dataset formatted to pass over to BAM/GAM modeling.
#'
env_format_fc <- function(env_data_extd,
                          quo_groupfield,
                          quo_obsfield){
  #turns long format into wide format - one entry per day per group
  #1: groupfield, 2: Date, 3: numericdate, 4+: env var (column name is env name)
  env_spread <- env_data_extd %>%
    dplyr::mutate(numericdate = as.numeric(.data$obs_date)) %>%
    dplyr::select(!!quo_groupfield, !!quo_obsfield, .data$obs_date, .data$numericdate, .data$val_epidemiar) %>%
    tidyr::spread(key = !!quo_obsfield, value = .data$val_epidemiar)

  env_spread
}

#'Format epi data for modeling
#'
#'@param epi_data_extd An epidemiological dataset extended into the
#'  future/forecast period with NA values for to-be-forecasted case numbers.
#'@param fc_clusters Extract from `report_settings$fc_clusters`, the
#'  geographic clusters to use in modeling.
#'
#'@inheritParams run_forecast
#'
#'@return An epidemiological dataset formatted to pass over to BAM/GAM modeling.
#'
epi_format_fc <- function(epi_data_extd,
                          quo_groupfield,
                          fc_clusters){

  #Get cluster information from model
  epi_format <- epi_data_extd %>%
    #join with cluster info
    dplyr::left_join(fc_clusters,
                     #NSE
                     by = rlang::set_names(rlang::as_name(quo_groupfield),
                                           rlang::as_name(quo_groupfield))) %>%
    #set cluster id as factor, must be for regression later
    dplyr::mutate(cluster_id = as.factor(.data$cluster_id),
                  #doy for cyclical regression
                  doy = as.numeric(format(.data$obs_date, "%j")),
                  #need numeric date for regression
                  numericdate = as.numeric(.data$obs_date))

  epi_format
}

#'Convert environmental data into anomalies.
#'
#'Raw environmental values are not used in modeling, but rather their anomalies,
#'departures for the historical "normal". We are looking at the influence of
#'deviation from normal in the environmental factors to help explain deviations
#'from normal in the human cases.
#'
#'@param env_fc Environmental data formatted for forecasting by
#'  `env_format_fc()`.
#'@param env_variables_used List of environmental variables that were used in
#'  the modeling, created by `pull_model_envvars()`, from list in
#'  `report_settings$env_var` & found in `env_data`
#'@param nthreads Extract from `report_settings$fc_nthreads`, max thread count
#'  for parallelization
#'
#'@inheritParams run_forecast
#'
#'@return Environmental dataset in same format as env_fc but with the residuals
#'  from a GAM with geographic unit and cyclical cubic regression spline on day
#'  of year per geographic group in place of the original raw values.
#'
anomalize_env <- function(env_fc,
                          quo_groupfield,
                          nthreads,
                          #internal/calculated
                          env_variables_used) {

  # Loop through each environmental variable replacing non-NA observations
  # with residuals from a gam with only geographic area (group) and day of year

  #Originally written with dataframes. Tibble/NSE/dplyr conversion not yet fully done.
  # #new tibble
  # env_fc_anom <- env_fc %>%
  #   mutate(group_factor = factor(!!quo_groupfield),
  #          doy = format(obs_date, "%j")) %>% as.numeric()

  # needed data for gam
  group_factor <- env_fc %>% dplyr::pull(!!quo_groupfield) %>% factor()
  doy <- env_fc %>% dplyr::pull(.data$obs_date) %>% format("%j") %>% as.numeric()

  env_fc <- as.data.frame(env_fc)

  # loop through environmental columns
  # note: brittle on column position until rewrite
  for (curcol in 4:ncol(env_fc)) {

    #if more than one geographic area
    if (nlevels(group_factor) > 1){
      tempbam <- mgcv::bam(env_fc[,curcol] ~ group_factor + s(doy, bs="cc", by=group_factor),
                           data=env_fc,
                           discrete = TRUE,
                           nthreads = nthreads)
    } else {
      #if only 1 geographic area, then run without group_factor
      tempbam <- mgcv::bam(env_fc[,curcol] ~ s(doy, bs="cc"),
                           data = env_fc,
                           discrete = TRUE,
                           nthreads = nthreads)
    }

    # could perhaps more cleverly be figured out by understanding the na.options of bam,
    # but for the moment just replace non-NA observations with their residuals
    env_fc[!is.na(env_fc[,curcol]),curcol] <- tempbam$residuals

  }

  env_fc <- tibble::as_tibble(env_fc)
  return(env_fc)

}

#'Lag the environmental data.
#'
#'@param epi_fc An epidemiological dataset extended into the future/forecast
#'  period with NA values for to-be-forecasted case numbers, as formatted for
#'  forecasting by epi_format_fc().
#'@param env_fc Environmental data formatted for forecasting by env_format_fc().
#'@param env_variables_used List of environmental variables that were used in
#'  the modeling, created by `pull_model_envvars()`, from list in
#'  `report_settings$env_var` & found in `env_data`
#'
#'@inheritParams run_forecast
#'
#'@return Wide dataset based on epidemiological data dates with five
#'  bandsummaries per environmental variable, from the basis spline summaries of
#'  the lagged environmental variable.
#'
lag_environ_to_epi <- function(epi_fc,
                               env_fc,
                               quo_groupfield,
                               report_settings,
                               #calculated/internal
                               groupings,
                               env_variables_used){

  lag_len <- report_settings[["env_lag_length"]]

  #create lag frame
  datalagger <- tidyr::crossing(group_temp = groupings,
                                obs_date = unique(epi_fc$obs_date),
                                lag = seq(from = 0, to = lag_len - 1, by = 1)) %>%
    #add lagging date
    dplyr::mutate(laggeddate = .data$obs_date - as.difftime(.data$lag, units = "days"))

  #and fix names with NSE
  datalagger <- datalagger %>%
    dplyr::rename(!!rlang::as_name(quo_groupfield) := .data$group_temp)

  #add env data
  datalagger <- dplyr::left_join(datalagger, env_fc,
                                 #because dplyr NSE, notice flip order
                                 by = rlang::set_names(c(rlang::as_name(quo_groupfield), "obs_date"),
                                                       c(rlang::as_name(quo_groupfield), "laggeddate")))

  # pivot lagged environmental data to epi data
  epi_lagged <- epi_fc #to more easily debug and rerun
  for (curcol in which(colnames(env_fc) %in% env_variables_used)){
    valuevar <- colnames(env_fc)[curcol]
    #wide data for all lags of that env var
    meandat <- datalagger %>%
      dplyr::select(!!quo_groupfield, .data$obs_date, .data$lag, valuevar) %>%
      tidyr::spread(key = .data$lag, value = valuevar)
    #rename lag columns (but not groupfield or Date)
    names(meandat)[-(1:2)] <- paste0(valuevar, "_", names(meandat)[-(1:2)])

    #join cur var wide data to epi data
    epi_lagged <- dplyr::left_join(epi_lagged, meandat,
                                   #dplyr NSE
                                   by = rlang::set_names(c(rlang::as_name(quo_groupfield), "obs_date"),
                                                         c(rlang::as_name(quo_groupfield), "obs_date")))
  } #end pivot loop

  #if using modified b-splines, do the basis functions and calcs here
  if (report_settings[["fc_splines"]] == "modbs"){
    # set up distributed lag basis functions (creates 7 basis functions)
    alpha <- 1/4
    distlagfunc <- splines::bs(x=seq(from=1, to=lag_len, by=1), intercept=TRUE,
                               knots=stats::quantile(seq(from=1, to=lag_len, by=1),
                                                     probs=seq(from=alpha, to=1-alpha, by=alpha),
                                                     na.rm=TRUE))
    dlagdeg <- ncol(distlagfunc)


    # create actual distributed lag summaries
    for (curvar in env_variables_used){
      bandsum <- matrix(data = rep(0, nrow(epi_lagged) * dlagdeg),
                        nrow = nrow(epi_lagged), ncol = dlagdeg)
      #first column of that variable (0 lag)
      mindex <- which(colnames(epi_lagged) == paste0(curvar, "_0"))
      #temp working matrix
      bandtemp <- as.matrix(epi_lagged[, (mindex:(mindex+lag_len-1))])
      #distributed lag summaries
      for (j in 1:dlagdeg){
        bandsum[, j] <- bandtemp %*% distlagfunc[,j]
      }
      bandsum <- data.frame(bandsum)
      names(bandsum) <- paste0("bandsum_", curvar, "_", 1:dlagdeg)

      # we used to do a submatrix here so that the regression formulae would
      # be more easily written, but this was incompatible with dplyr
      epi_lagged <- dplyr::bind_cols(epi_lagged, bandsum)

      #created summary value for each basis function (5) per env variable per group per week (based on epidemiological data time unit)

    } #end distr lag summary loop

    #only keep bandsummaries (daily lags can be removed to free up a lot of space)
    #  note: ^ matches beginning of string, otherwise we'd get the bandsummaries too, which we want to keep
    for (cvar in env_variables_used){
      epi_lagged[, which(grepl(paste0("^", cvar, "_"), colnames(epi_lagged)))] <- NULL
    }
  }

  #return
  epi_lagged
}


#' Creates a modified b-spline basis (piecewise polynomial).
#'
#' The modified basis splines are used to capture any long term trends per
#' geographic group.
#'
#'@param x Vector of weekly observation dates.
#'@param degree Degree passed to splines::bs().
#'@param maxobs Date of the last known value.
#'@param minobs Date of the first known value.
#'
#'@return A modified b-spline basis with the last basis splines reversed and
#'  the second to last basis spline function removed (to reduce the edge effects
#'  of using splines).
#'
truncpoly <- function(x = NULL, degree = 6, maxobs = NULL, minobs = NULL){

  # Some of the later functions will convert date to type spline, so
  # it's best to go ahead and convert now. Left_join doesn't convert
  # dates to numeric on the fly.
  x <- as.numeric(x)

  # create frame to hold modified b-spline basis
  xdf  <- data.frame(x=x)

  # figure out where we apparently have data
  apparentminobs <- min(xdf$x, na.rm=TRUE)
  apparentmaxobs <- max(xdf$x, na.rm=TRUE)

  # figure out where the bspline basis will have support
  if (!is.null(minobs)) {

    actualminobs <- max(apparentminobs, minobs)

  } else { actualminobs <- apparentminobs }
  if (!is.null(maxobs)) {

    actualmaxobs <- min(apparentmaxobs, maxobs)

  } else { actualmaxobs <- apparentmaxobs }

  # create a full frame to hold this basis before truncation
  xdf2     <- data.frame(x = actualminobs:actualmaxobs)
  xdf2$bas <- splines::bs(x=xdf2$x, degree=degree)

  # reverse the last spline basis function
  xdf2$bas[,degree] <- rev(xdf2$bas[,degree])

  # delete the next to last spline basis function
  xdf2$bas <- xdf2$bas[,-(degree-1)]

  # merge with original frame
  xdf <- dplyr::left_join(xdf, xdf2, by="x")

  # make values 0 where we extend beyond actualmax/minobs
  for (colnum in 1:ncol(xdf$bas)) {

    xdf$bas[is.na(xdf$bas[,colnum]),colnum] <- 0

  }

  tempdf <- data.frame(xdf$bas)
  names(tempdf) <- paste("modbs_reserved_", names(tempdf), sep="")

  return(tempdf)

}


#'Formats the environmental data lagged to epidemiology data the way that the
#'clusterapply package wants.
#'
#'@param tbl The tibble with all the lagged variables wide and flat.
#'@param env_variables_used Vector of the names of the environmental variables
#'  that are being used in the model.
#'
#'@inheritParams run_forecast
#'
#'@return A dataframe with sub-matrices for each of the lagged environmental
#'  variable data.
#'
format_lag_ca <- function(tbl,
                          env_variables_used,
                          report_settings){

  #initialize
  #vector to collect all lagged environmental column names
  all_lag_cols <- vector()
  #dataframe to collect the lagged environmental variables as sub matrices
  collecting_df <- as.data.frame(matrix(nrow = nrow(tbl), ncol = length(env_variables_used)))

  #loop for each environmental variable used in modeling
  for (v in seq_along(env_variables_used)){
    cur_var <- env_variables_used[[v]]
    #column names are {env_var}_n, where n is the lag day
    var_allcol <- grep(paste(cur_var,"*"), colnames(tbl), value = TRUE)
    #append column names to master list
    all_lag_cols <- c(all_lag_cols, var_allcol)

    #create a matrix of just that environmental variable
    var_mat <- tbl %>%
      dplyr::select(var_allcol) %>%
      as.matrix()
    #put into collecting dataframe
    collecting_df[,v] <- var_mat
    #name column as the variable
    names(collecting_df)[v] <- cur_var
  }

  #get columns that are NOT lagged environmental variables
  front_df <- tbl %>%
    dplyr::select(-all_lag_cols) %>%
    as.data.frame()

  #create lag matrix (days of lag)
  #next column after vars
  index_lag <- length(env_variables_used) + 1
  collecting_df[,index_lag] <- matrix(data = rep(0:(report_settings[["env_lag_length"]]-1),
                                                 times = nrow(tbl)),
                                      nrow = nrow(tbl),
                                      ncol = report_settings[["env_lag_length"]],
                                      byrow = TRUE)
  colnames(collecting_df[,index_lag]) <- 0:(report_settings[["env_lag_length"]] - 1)
  names(collecting_df)[index_lag] <- "lag"

  #column bind the non-lagged with the submatrix-filled dataframe
  dfm <- cbind(front_df, collecting_df)

  ## Also add censored numericdate -
  # this prevents splines from going off into extreme directions when forecasting into the future
  numericdate_end_known <- dfm %>%
    dplyr::filter(.data$input == 1) %>%
    dplyr::summarize(maxdt = max(.data$numericdate, na.rm = TRUE)) %>%
    dplyr::pull(.data$maxdt)
  #censored date is the numericdate, but capped at the latest known data
  dfm <- dfm %>%
    dplyr::mutate(censored_date = pmin(.data$numericdate, numericdate_end_known))

  #return
  dfm
}

#'Checks that all models successfully built when using batch_bam.
#'
#'@param reg_bb The regression model(s) returned by batch_bam.
#'
#'@return None. Will stop will informative error message if
#'problems are found.
#'
check_bb_models <- function(reg_bb){

  #check if each is a bam model
  is_bam <- lapply(reg_bb, methods::is, "bam")

  #if ALL are not bam models, then stop and return error messages(s) from modeling
  if (!all(unlist(is_bam))){

    #get which failed
    fails <- names(is_bam[sapply(is_bam, function(x) x[1]==FALSE)])

    #quick data frame for failure model names and messages
    fails_msg_df <- data.frame(Model = unlist(names(reg_bb[fails])),
                              Error_message = unlist(lapply(reg_bb[fails],
                                                  function(x) conditionMessage(x))))

    #stop and message
    stop(paste("Model(s) failed, please review: \n",
               paste0(utils::capture.output(fails_msg_df), collapse = "\n")))

  }
}



