

#'Functions to check input to epidemiar and set report settings defaults.
#'
#'Function does basic existance checks and variety of logic checks on input data
#'to run_epidemia(), and sets defaults to report_settings parameters.
#'
#'@param quo_casefield Quosure of user given field containing the disease case
#'  counts.
#'@param quo_popfield Quosure of user-given field containing population values.
#'@param quo_groupfield Quosure of the user given geographic grouping field to
#'  run_epidemia().
#'@param quo_obsfield Quosure of user given field name of the environmental data
#'  variables
#'@param quo_valuefield Quosure of user given field name of the value of the
#'  environmental data variable observations.
#'@param raw_settings The report_settings object as given by the user.
#'@param groupings List of all unique geographical groupings in epi_data.
#'@param env_variables List of all unique environmental variables in env_data.
#'
#'@inheritParams run_epidemia
#'
#'@return Returns a list of items: a flag if there were any errors, plus accompanying error
#'  messages; a flag and messages for warnings; updated report_settings
#'
#'
#'

input_check <- function(epi_data,
                        env_data,
                        env_ref_data,
                        env_info,
                        quo_casefield,
                        quo_popfield,
                        quo_groupfield,
                        quo_obsfield,
                        quo_valuefield,
                        fc_model_family,
                        raw_settings,
                        groupings,
                        env_variables){

  #input checks and setting defaults have logic interwoven, so are both handled here.
  # in general, check given setting and copy over if pass, otherwise if missing, assign default
  # this will double-check that default values are good (avoid possible unexpected situations)

  # Want ALL data checks to happen, whether or not error happen before the end of the tests.
  # Want to collect all errors, and return all of them to console
  # Exception: if some early/critical errors happen, it will need to be returned early or skip certain later checks that would fail to check properly.
  # Note: does not test for integer value (versus simply numeric), which is non-trivial, for various items.

  # Create err_flag (binary if any error) and err_msgs (all error messages) variables
  err_flag <- FALSE
  err_msgs <- ""

  # Also collect any warning messages to display
  warn_flag <- FALSE
  warn_msgs <- "Warning messages:\n"

  #set up cleaned list
  new_settings <- list()


  # 1. Required fields checking --------------------------------------------------------

  #Confirm that user supplied field names exists in datasets

  # epi_data tests
  # has obs_date as Date
  if (!"obs_date" %in% colnames(epi_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column 'obs_date' in the epidemiological dataset, 'epi_data'.\n")
  } else if (!class(epi_data$obs_date) == "Date"){
    #has obs_date, now check type
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "The 'obs_date' field in the epidemiological dataset, 'epi_data', must be type Date.\n")
  }
  # has casefield
  if (!rlang::as_name(quo_casefield) %in% colnames(epi_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_casefield), ", in the epidemiological dataset, 'epi_data'.\n")
  }
  # has groupfield
  if(!rlang::as_name(quo_groupfield) %in% colnames(epi_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_groupfield), ", in the epidemiological dataset, 'epi_data'.\n")
  }
  # has populationfield, but only if given as it is optional
  #testing if quosure was created on NULL object.
  if(!rlang::quo_is_null(quo_popfield)){
    if(!rlang::as_name(quo_popfield) %in% colnames(epi_data)){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "The specified column ", rlang::as_name(quo_popfield), ", for population must be in the in the epidemiological dataset, 'epi_data'.\n")
    }
  }


  # env_data tests
  # has obs_date as Date
  if(!"obs_date" %in% colnames(env_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column 'obs_date' in the environmental dataset, 'env_data'.\n")
  } else if(!class(env_data$obs_date) == "Date"){
    #has obs_date, now check type
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "The 'obs_date' field in the environmental dataset, 'env_data', must be type Date.\n")
  }
  # has groupfield
  if(!rlang::as_name(quo_groupfield) %in% colnames(env_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_groupfield), ", in the environmental dataset, 'env_data'.\n")
  }
  # has obsfield
  if(!rlang::as_name(quo_obsfield) %in% colnames(env_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_obsfield), ", in the environmental dataset, 'env_data'.\n")
  }
  # has valuefield
  if(!rlang::as_name(quo_valuefield) %in% colnames(env_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_valuefield), ", in the environmental dataset, 'env_data'.\n")
  }


  # env_ref tests
  # has groupfield
  if(!rlang::as_name(quo_groupfield) %in% colnames(env_ref_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_groupfield), ", in the environmental reference dataset, 'env_ref_data'.\n")
  }
  # has obsfield
  if(!rlang::as_name(quo_obsfield) %in% colnames(env_ref_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_obsfield), ", in the environmental reference dataset, 'env_ref_data'.\n")
  }
  #has week_epidemiar
  if(!"week_epidemiar" %in% colnames(env_ref_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column 'week_epidemiar' for the week of the year in the environmental reference dataset, 'env_ref_data'.\n")
  } else if(!(is.numeric(env_ref_data$week_epidemiar) | is.integer(env_ref_data$week_epidemiar))){
    #week_epidemiar exists, now check class/type
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "The column 'week_epidemiar' in 'env_ref_data' must be numeric or integer type (integer values only).\n")
  }
  #has ref_value
  if(!"ref_value" %in% colnames(env_ref_data)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column 'ref_value' for the historical reference value in the dataset, 'env_ref_data'.\n")
  }

  # env_info
  # has obsfield
  if(!rlang::as_name(quo_obsfield) %in% colnames(env_info)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_obsfield), ", in the environmental metadata file, 'env_info'.\n")
  }
  # has reference_method
  if(!"reference_method" %in% colnames(env_info)){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "There must be a column 'reference_method' in 'env_info' for how to summarize values from daily to weekly ('sum' or 'mean').\n")
  }

  #STOP early if errors by now

  #if errors, stop and return error messages
  if (err_flag){
    #prevent possible truncation of all error messages
    options(warning.length = 4000L)
    stop(err_msgs)
  }


  # 2. Report dates and lengths --------------------------------------------------------

  #special flag for all length periods
  rpt_len_flag <- FALSE


  #fc_start_date: date when to start forecasting
  if (!is.null(raw_settings[["fc_start_date"]])){
    #check that date type
    if (!class(raw_settings[["fc_start_date"]]) == "Date"){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$fc_start_date' must be type Date.\n")
    } else {
      #copy over checked user value
      new_settings[["fc_start_date"]] <- raw_settings[["fc_start_date"]]
    }
  } else {
    # defaults to last known epidemiological data date + one week
    last_known <- max(epi_data[["obs_date"]], na.rm = TRUE)
    new_settings[["fc_start_date"]] <- last_known + lubridate::as.difftime(1, units = "weeks")
    #Removed warning message, this is a good default / normal setting
    #warn_flag <- TRUE
    #warn_msgs <- paste0(warn_msgs, "'report_settings$fc_start_date' was not provided, running with default ", new_settings[["fc_start_date"]], ".\n")
  }

  #report_period
  if (!is.null(raw_settings[["report_period"]])){
    #check type
    if (!(is.numeric(raw_settings[["report_period"]]) | is.integer(raw_settings[["report_period"]]))){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$report_period' must be numeric or integer type - integer number of weeks only.\n")
      rpt_len_flag <- TRUE
    } else {
      #copy over checked user value
      new_settings[["report_period"]] <- raw_settings[["report_period"]]
    }
  } else {
    #default
    new_settings[["report_period"]] <- 26
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_settings$report_period' was not provided, running with default ", new_settings[["report_period"]], ".\n")
  }

  #ed_summary_period
  if (!is.null(raw_settings[["ed_summary_period"]])){
    if (!(is.numeric(raw_settings[["ed_summary_period"]]) | is.integer(raw_settings[["ed_summary_period"]]))){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$ed_summary_period' must be numeric or integer type - integer number of weeks only.\n")
      rpt_len_flag <- TRUE
    } else {
      #copy over checked user value
      new_settings[["ed_summary_period"]] <- raw_settings[["ed_summary_period"]]
    }
  } else {
    #default
    new_settings[["ed_summary_period"]] <- 4
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_settings$ed_summary_period' was not provided, running with default ", new_settings[["ed_summary_period"]], ".\n")

  }

  #fc_future_period
  if (!is.null(raw_settings[["fc_future_period"]])){
    if (!(is.numeric(raw_settings[["fc_future_period"]]) | is.integer(raw_settings[["fc_future_period"]]))){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$forecast_future' must be numeric or integer type - integer number of weeks only.\n")
      rpt_len_flag <- TRUE
    } else {
      #copy over checked user value
      new_settings[["fc_future_period"]] <- raw_settings[["fc_future_period"]]
    }
  } else {
    #default
    new_settings[["fc_future_period"]] <- 8
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_settings$fc_future_period' was not provided, running with default ", new_settings[["fc_future_period"]], ".\n")

  }

  #if none of the user entered lengths throw an error, now continue with testing override settings if needed
  if (!rpt_len_flag){
    #report lengths structure:
    # full report length must be at least 1 time unit longer than forecast period + any ed summary period
    # (will also handle: ed summary period must be <= time points than 'prev' period (report length - forecast length))
    if (new_settings[["report_period"]] < new_settings[["fc_future_period"]] + min(1, new_settings[["ed_summary_period"]])) {
      #make report period make sense with forecast period and (possible) ed summary period
      new_settings[["report_period"]] <- new_settings[["fc_future_period"]] + max(1, new_settings[["ed_summary_period"]])
      warn_flag <- TRUE
      warn_msgs <- paste0(warn_msgs, "With forecast period ", new_settings[["fc_future_period"]],
              " and event detection summary period ", new_settings[["ed_summary_period"]],
              ", the report length has been adjusted to ", new_settings[["report_period"]], ".\n")
    }
  }



  # 3. Forecasting ---------------------------------------------------------------------

  #fc_cyclicals
  if (!is.null(raw_settings[["fc_cyclicals"]])){
    #skipping trying to test for boolean given R
    #copy
    new_settings[["fc_cyclicals"]] <- raw_settings[["fc_cyclicals"]]
  } else {
    #default
    new_settings[["fc_cyclicals"]] <- FALSE
  }


  #fc_cyclicals_by 'group' or 'cluster'

  # if provided, prepare for matching
  if (!is.null(raw_settings[["fc_cyclicals_by"]])){
    new_settings[["fc_cyclicals_by"]] <- tolower(raw_settings[["fc_cyclicals_by"]])
  } else {
    #if not provided/missing/null
    new_settings[["fc_cyclicals_by"]] <- "cluster"
  }
  #try match
  new_settings[["fc_cyclicals_by"]] <- tryCatch({
    match.arg(new_settings[["fc_cyclicals_by"]], c("cluster", "group"))
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "Given 'fc_cyclicals_by'",
                        raw_settings[["fc_cyclicals_by"]],
                        "does not match 'cluster' or 'group', running as 'cluster'.\n")
    "cluster"
  })



  #env_var
  #has entries in env_data, env_ref_data, & env_info?
  #create list of all environmental variables in env_info
  env_info_variables <- dplyr::pull(env_info, !!quo_obsfield) %>% unique()
  #create list of all environmental variables in env_ref_data
  env_ref_variables <- dplyr::pull(env_ref_data, !!quo_obsfield) %>% unique()
  #env_variables already gen list of env_data

  if (!is.null(raw_settings[["env_var"]])){
    # given env_var
    # check has obsfield
    if(!rlang::as_name(quo_obsfield) %in% colnames(raw_settings[["env_var"]])){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "There must be a column", rlang::as_name(quo_obsfield),
                        ", to indicate the list of model environmental variables in 'report_settings$env_vars'.\n")
    } else {
      #does have obsfield,
      #check that model variables exist in env data and env ref data
      #special flag for env var existing
      env_var_flag <- FALSE

      #pull variables from model info input
      model_vars <- raw_settings[["env_var"]] %>% dplyr::pull(!!quo_obsfield)

      if (!all(model_vars %in% env_variables)){
        env_var_flag <- TRUE
        err_flag <- TRUE
        err_msgs <- paste0(err_msgs, "Model variable(s) given in 'report_settings$env_var' is/are missing from 'env_data':\n",
                          model_vars[which(!model_vars %in% env_variables)], ".\n")
      }
      if (!all(model_vars %in% env_ref_variables)){
        env_var_flag <- TRUE
        err_flag <- TRUE
        err_msgs <- paste0(err_msgs, "Model variable(s) given in 'report_settings$env_var' is/are missing from 'env_ref_data': ",
                          model_vars[which(!model_vars %in% env_ref_variables)], "\n")
      }
      if (!all(model_vars %in% env_info_variables)){
        env_var_flag <- TRUE
        err_flag <- TRUE
        err_msgs <- paste0(err_msgs, "Model variable(s) given in 'report_settings$env_var' is/are missing from 'env_info': ",
                          model_vars[which(!model_vars %in% env_info_variables)], "\n")
      }
      if (!env_var_flag){
        #if passed checks, copy
        new_settings[["env_var"]] <- raw_settings[["env_var"]]
      }
    } #end else obsfield
  } else {
    #default
    #Two sets of intersection to create list that are present in all three
    env_data_info <- dplyr::intersect(env_variables, env_info_variables)
    default_env_var <- dplyr::intersect(env_data_info, env_ref_variables)
    new_settings[["env_var"]] <- dplyr::tibble(obs_temp = default_env_var) %>%
      #rename NSE fun
      dplyr::rename(!!rlang::as_name(quo_obsfield) := .data$obs_temp)
    #message result
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "No user supplied list of environmetal variables to use. Using: ",
                        paste(unlist(default_env_var), collapse = " "),
                        " based on presence in env_data, env_ref_data, and env_info.\n")
  }

  #fc_clusters
  if (!is.null(raw_settings[["fc_clusters"]])){
    #given clusters
    # special cluster flag
    cluster_flag <- FALSE
    # has groupfield
    if(!rlang::as_name(quo_groupfield) %in% colnames(raw_settings[["fc_clusters"]])){
      cluster_flag <- TRUE
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "There must be a column ", rlang::as_name(quo_groupfield),
                        ", in 'report_settings$clusters'.\n")
    }
    # has cluster_id
    if(!"cluster_id" %in% colnames(raw_settings[["fc_clusters"]])){
      cluster_flag <- TRUE
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "There must be a column 'cluster_id' in 'report_settings$clusters'.\n")
    }
    #now check that all geographic groupings from epi data have a cluster assigned
    #as long as no previous errors
    if (!cluster_flag){
      #groupings in cluster info
      model_cl <- raw_settings[["fc_clusters"]] %>% dplyr::pull(!!quo_groupfield)
      #groupings in epidemiological data
      groups_epi <- dplyr::pull(epi_data, !!quo_groupfield) %>% unique()
      #check all in cluster list
      if (!all(groups_epi %in% model_cl)){
        cluster_flag <- TRUE
        err_flag <- TRUE
        err_msgs <- paste0(err_msgs, "Geographic groupings present in the epidemiological data are missing in 'report_settings$clusters': ",
                          groups_epi[which(!groups_epi %in% model_cl)],
                          ".\n")
      }
      #Don't need to check environmental data too.
      #Extra env data for other groupings not in epidemiological data are just ignored.
    }
    if(!cluster_flag){
      #if passed checks, copy
      new_settings[["fc_clusters"]] <- raw_settings[["fc_clusters"]]
    }
  } else {
    #default
    #default is one cluster, probably not what you actually want for any type of large system
    #create tbl of only one cluster
    #groupings already exist as list of geographic groups
    cluster_tbl <- tibble::tibble(group_temp = groupings, cluster_id = 1) %>%
      #and fix names with NSE
      dplyr::rename(!!rlang::as_name(quo_groupfield) := .data$group_temp)
    #assign
    new_settings[["fc_clusters"]] <- cluster_tbl
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_settings$fc_clusters' was not provided, running with default of one cluster, i.e. a global model.\n")

  }


  #env_lag_length
  if (!is.null(raw_settings[["env_lag_length"]])){
    if (!(is.numeric(raw_settings[["env_lag_length"]]) | is.integer(raw_settings[["env_lag_length"]]))){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$env_lag_length' must be an integer number of days only.\n")
    } else {
      #copy over checked user value
      new_settings[["env_lag_length"]] <- raw_settings[["env_lag_length"]]
    }
  } else {
    #default
    #maybe make default based on data length, but for now
    new_settings[["env_lag_length"]] <- 181
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_settings$env_lag_length' was not provided, running with default ", new_settings[["env_lag_length"]], ".\n")

  }


  #has enough environmental data for lag length?
  #already checked existance and numeric/integer type
  #check that enough environmental data exists for lag length selected
  #but only if no other problems so far (would include env_var issues if found above)
  if (!err_flag){
    #subset to env variables as dictated by the model
    env_model_data <- pull_model_envvars(env_data, quo_obsfield, env_var = new_settings$env_var)
    #get earliest dates available
    env_start_dts <- env_model_data %>% dplyr::group_by(!!quo_obsfield) %>% dplyr::summarize(start_dt = min(.data$obs_date))
    #date needed by laglength and first epidemiological data date
    lag_need_dt <- min(epi_data$obs_date) - as.difftime(new_settings[["env_lag_length"]], units = "days")
    #all env dates equal or before needed date?
    if (!all(env_start_dts$start_dt <= lag_need_dt)){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "Not enough environmental data for a lag length of ",
                         new_settings[["env_lag_length"]],
                        "days.\n Epidemiological start is", min(epi_data$obs_date),
                        "therefore environmental data is needed starting", lag_need_dt,
                        "for variables:",
                        env_start_dts[which(!env_start_dts$start_dt <= lag_need_dt),1],
                        ".\n")

    }
  } #end err_flag

  #env_data: test for missing rows pre-report period
  # in report period (incl. 'future'), missing implicit/explicit will be handled by env filler/extender
  if (!err_flag){
    report_start_date <- new_settings[["fc_start_date"]] -
      lubridate::as.difftime((new_settings[["report_period"]] -
                                new_settings[["fc_future_period"]]),
                             unit = "weeks")
    pre_env_check <- env_data %>%
      #only pre-report data check
      dplyr::filter(.data$obs_date < report_start_date) %>%
      #and only after needed date for lag length (earlier entries don't matter)
      dplyr::filter(.data$obs_date >= lag_need_dt) %>%
      #field for error message
      dplyr::mutate(group_obs = paste0(!!quo_groupfield, "-", !!quo_obsfield)) %>%
      #calc number of rows, should be the same for all if no missing rows
      dplyr::group_by(.data$group_obs) %>%
      dplyr::summarize(rowcount = dplyr::n())
    not_max_env_rows <- pre_env_check %>%
      dplyr::filter(.data$rowcount < max(pre_env_check$rowcount))
    if (nrow(not_max_env_rows) > 1) {
      #some implicit missing rows
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "Missing rows detected in environmental data prior to report start date. ",
                         "Implicit missing data is not allowed, please add rows with NA values. ",
                         "Please check the following: ",
                         paste(unlist(dplyr::pull(not_max_env_rows, .data$group_obs)), collapse = " "),
                         ".\n")
    } #end if nrow > 1
  } #end if err_flag


  #fc_splines
  #is batchapply installed & available?
  batchbam_ok <- if (requireNamespace("clusterapply", quietly = TRUE)) {TRUE} else {FALSE}
  #if batchapply is installed then default is thin plate
  default_splines <- if (batchbam_ok) {'tp'} else {'modbs'}

  #check input
  if (!is.null(raw_settings[["fc_splines"]])) {
    #prep user input for matching
    new_settings[["fc_splines"]] <- tolower(raw_settings[["fc_splines"]])
  } else {
    #no user input, use default
    new_settings[["fc_splines"]] <- default_splines
  }
  #try match
  new_settings[["fc_splines"]] <- tryCatch({
    match.arg(new_settings[["fc_splines"]], c("modbs", "tp"))
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "Given 'fc_splines'",
                        raw_settings[["fc_splines"]],
                        "does not match 'modbs' or 'tp', running as ",
                        default_splines, ".\n")
    default_splines
  })
  #stop/error if requested tp if batchapply is not installed/available
  if (new_settings[["fc_splines"]] == "tp" & !batchbam_ok){
    err_flag <- TRUE
    err_msgs <- paste0(err_msgs, "User requested thin plate splines (fc_splines = 'tp'),",
                        "but package clusterapply is not installed/available. ",
                        "Try running with modified b-splines ('modbs') instead.\n")
  }


  #ncores
  if (!is.null(raw_settings[["fc_ncores"]])) {
    new_settings[["fc_ncores"]] <- raw_settings[["fc_ncores"]]
  } else {
    #calc default
    #detectCores can return NA, so catch
    new_settings[["fc_ncores"]] <- max(parallel::detectCores(logical=FALSE),
                                       1,
                                       na.rm = TRUE)
  }
  #nthreads
  if (!is.null(raw_settings[["fc_nthreads"]])) {
    # allow override
    new_settings[["fc_nthreads"]] <- raw_settings[["fc_nthreads"]]
  } else {
    #calc default: number of physical cores
    new_settings[["fc_nthreads"]] <- new_settings[["fc_ncores"]]
  }


  # Developer options
  if (!is.null(raw_settings[["dev_fc_fit_freq"]])){
    new_settings[["dev_fc_fit_freq"]] <- raw_settings[["dev_fc_fit_freq"]]
  } else {
    #default
    new_settings[["dev_fc_fit_freq"]] <- "once"
  }
  # for dev formula: dev must also set fc_splines and fc_cyclicals (if modbs) correctly,
  # otherwise it will not know which function to call
  # also need to set correct env vars (or let take all)
  if (!is.null(raw_settings[["dev_fc_formula"]])){
    new_settings[["dev_fc_formula"]] <- raw_settings[["dev_fc_formula"]]
  } else {
    #default
    new_settings[["dev_fc_formula"]] <- NULL
  }


  # 4. Report settings -----------------------------------------------------------------

  #epi_interpolate
  if (!is.null(raw_settings[["epi_interpolate"]])){
    #skipping trying to test for boolean given R
    #copy
    new_settings[["epi_interpolate"]] <- raw_settings[["epi_interpolate"]]
  } else {
    #default
    new_settings[["epi_interpolate"]] <- FALSE
  }


  #env_anomalies
  if (!is.null(raw_settings[["env_anomalies"]])){
    #skipping trying to test for boolean given R
    #copy
    new_settings[["env_anomalies"]] <- raw_settings[["env_anomalies"]]
  } else {
    #default
    new_settings[["env_anomalies"]] <- dplyr::case_when(
      #being very explicit to make sure in naive models this is false
      fc_model_family == "naive-persistence" ~ FALSE,
      fc_model_family == "naive-weekaverage" ~ FALSE,
      #default to FALSE
      TRUE ~ FALSE)
  }


  #report_inc_per
  if (!is.null(raw_settings[["report_inc_per"]])){
    if (!is.numeric(raw_settings[["report_inc_per"]]) || raw_settings[["report_inc_per"]] <= 0){
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "'report_settings$report_inc_per' must be numeric and a positive number.\n")
    } else {
      #copy
      new_settings[["report_inc_per"]] <- raw_settings[["report_inc_per"]]
    }
  } else {
    #default
    new_settings[["report_inc_per"]] <- 1000
  }


  # For things that are being string matched:
  # tolower to capture upper and lower case user-input variations since match.arg is case sensitive
  # but must only try function if ed_method is not null (i.e. was given)

  #report_value_type
  # if provided, prepare for matching
  if (!is.null(raw_settings[["report_value_type"]])){
    new_settings[["report_value_type"]] <- tolower(raw_settings[["report_value_type"]])
  } else {
    #if not provided/missing/null
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'report_value_type' was not provided, returning results in case counts ('cases').\n")
    new_settings[["report_value_type"]] <- "cases"
  }
  #try match
  new_settings[["report_value_type"]] <- tryCatch({
    match.arg(new_settings[["report_value_type"]], c("cases", "incidence"))
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "Given 'report_value_type'",
                        raw_settings[["report_value_type"]],
                        "does not match 'cases' or 'incidence', running as 'cases'.\n")
    "cases"
  })

  # epi_date_type
  # if provided, prepare for matching
  if (!is.null(raw_settings[["epi_date_type"]])){
    #want to keep ISO and CDC capitalized, but drop 'Week' to 'week' if had been entered that way
    first_char <- substr(raw_settings[["epi_date_type"]], 1, 1) %>%
      tolower()
    #remainder of user entry
    rest_char <- substr(raw_settings[["epi_date_type"]], 2, nchar(raw_settings[["epi_date_type"]]))
    #paste back together
    new_settings[["epi_date_type"]] <- paste0(first_char, rest_char)
  } else {
    #if not provided/missing/null
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "'epi_date_type' was not provided, running as weekly, ISO/WHO standard ('weekISO').\n")
    new_settings[["epi_date_type"]] <- "weekISO"
  }
  #try match
  new_settings[["epi_date_type"]] <- tryCatch({
    match.arg(new_settings[["epi_date_type"]], c("weekISO", "weekCDC")) #"monthly" reserved for future
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "Given 'epi_date_type'", raw_settings[["epi_date_type"]],
                        "does not match 'weekISO' or 'weekCDC', running as 'weekISO' (weekly, ISO/WHO standard).\n")
    "weekISO"
  })

  #epi_transform
  # if provided, prepare for matching
  if (!is.null(raw_settings[["epi_transform"]])){
    new_settings[["epi_transform"]] <- tolower(raw_settings[["epi_transform"]])
  } else {
    #if not provided/missing/null
    #nothing checks in case it in "none", but set for clarity, esp. in metadata
    new_settings[["epi_transform"]] <- "none"
  }
  #try match
  new_settings[["epi_transform"]] <- tryCatch({
    match.arg(new_settings[["epi_transform"]], c("none", "log_plus_one"))
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs, "Given 'epi_transform'", raw_settings[["epi_transform"]],
                        "does not match 'none' or 'log_plus_one', running as 'none'.\n")
    "none"
  })



  # 5. Early Detection settings --------------------------------------------------------

  # For things that are being string matched:
  # tolower to capture upper and lower case user-input variations since match.arg is case sensitive
  # but must only try function if ed_method is not null (i.e. was given)

  # ed_method
  # if provided, prepare for matching
  if (!is.null(raw_settings[["ed_method"]])){
    new_settings[["ed_method"]] <- tolower(raw_settings[["ed_method"]])
  } else {
    #if not provided/missing/null
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs,"'ed_method' was not provided, running as 'none'.\n")
    new_settings[["ed_method"]] <- "none"
  }
  #try match
  new_settings[["ed_method"]] <- tryCatch({
    match.arg(new_settings[["ed_method"]], c("none", "farrington"))
  }, error = function(e){
    warn_flag <- TRUE
    warn_msgs <- paste0(warn_msgs,"Given 'ed_method' ", raw_settings[["ed_method"]],
                        " does not match 'none' or 'farrington', running as 'none'.\n")
    "none"
  })

  #ed_control
  if (!is.null(raw_settings[["ed_control"]])){
    #just copy over, no testing here
    new_settings[["ed_control"]] <- raw_settings[["ed_control"]]
  }


  #special check/message for Farrington
  if (new_settings[["ed_method"]] == "farrington"){

    #controls for Farrington all have defaults in farringtonFlexible() and can be missing, just warn
    if (is.null(raw_settings[["ed_control"]])){
      #warning if missing though
      warn_flag <- TRUE
      warn_msgs <- paste(warn_msgs, "Early Detection controls not found, running with surveillance package defaults.\n")
    }
  }


  # 6. Model runs and caching -----------------------------------------------------------

  #model_run
  if (!is.null(raw_settings[["model_run"]])){
    #copy over (skipping boolean check because R)
    new_settings[["model_run"]] <- raw_settings[["model_run"]]
  } else {
    #default
    new_settings[["model_run"]] <- FALSE
  }

  #model_cached
  if (!is.null(raw_settings[["model_cached"]])){
    #check that $model_info and $model_obj exists in model_cached
    if (all(c("model_obj", "model_info") %in% names(raw_settings[["model_cached"]]))){

      #if model looks okay so far, then check further

      #Removed: batch_bam returns list of models, will need to reconsider how to test
      # #make sure given model (if given) is a regression object (using basic "lm" as test)
      # #model_cached$model_obj
      # classes <- class(raw_settings[["model_cached"]][["model_obj"]])
      # if (!"lm" %in% classes){
      #   err_flag <- TRUE
      #   err_msgs <- paste0(err_msgs, "The object in 'report_settings$model_cached$model_obj' is not a regression object, found classes are: ", classes, ".\n")
      # } #end lm check

      #if using a cached model, the model family from the cached model will be used
      #warn about overriding any user input family
      if ((fc_model_family != "cached") &
          (raw_settings$model_cached$model_info$fc_model_family != fc_model_family)){
        warn_flag <- TRUE
        warn_msgs <- paste0(warn_msgs, "The cached model family ", raw_settings$model_cached$model_info$fc_model_family, ", will override any user input. ",
                           "Found 'fc_model_family' set to ",  fc_model_family, " instead of 'cached'.\n")
      }

      #use metadata to override fc_splines if needed. This MUST match for the correct function to be called.
      if (raw_settings$model_cached$model_info$report_settings$fc_splines != new_settings$fc_splines){
        warn_flag <- TRUE
        warn_msgs <- paste0(warn_msgs, "The cached model fc_splines ",
                            raw_settings$model_cached$model_info$report_settings$fc_splines,
                            ", will override report_settings$fc_splines: ",
                            new_settings$fc_splines)
        new_settings$fc_splines <- raw_settings$model_cached$model_info$report_settings$fc_splines
        #and repeat test that batch_bam is ok
        #stop/error if requested tp if batchapply is not installed/available
        if (new_settings[["fc_splines"]] == "tp" & !batchbam_ok){
          err_flag <- TRUE
          err_msgs <- paste0(err_msgs, "Cached model uses thin plate splines (fc_splines = 'tp'),",
                             "but package clusterapply is not installed/available. \n")
        }

      }

      #end if names
    } else {
      err_flag <- TRUE
      err_msgs <- paste0(err_msgs, "The given cached model is missing $model_obj and/or $model_info.\n")
    } #end else on if names

    #copy model over
    new_settings[["model_cached"]] <- raw_settings[["model_cached"]]

  } else {
    #default
    new_settings[["model_cached"]] <- NULL
  }


  #check if fc_model_family is cached that a cached model was given, else fail with error
  if (fc_model_family == "cached"){
    if (is.null(new_settings[["model_cached"]])){
      err_flag <- TRUE
      err_msgs <- paste(err_msgs, "If 'fc_model_family' is set to 'cached', a cached model must be supplied in 'report_settings$model_cached'.\n")
    }
  }



  # Return -----------------------------------------------------------

  ## Return
  create_named_list(err_flag, err_msgs, warn_flag, warn_msgs, clean_settings = new_settings)

} #end input_check()


