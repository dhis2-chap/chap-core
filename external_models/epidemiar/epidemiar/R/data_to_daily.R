#' Converts non-daily data to daily data, with optional interpolation
#'
#' Takes in a dataset with implicit missing data (missing rows) or interval data
#' and expands it into daily data, filling in with NAs, and then optionally
#' linearly interpolating all NA values.
#'
#' @param data_notdaily Input dataset with values in valuefield, and date field
#'   in "obs_date". Function will group on all other fields present and create
#'   combinations of those plus the date.
#' @param valuefield Field that the observation values are in.
#' @param interpolate Logical to whether or not interpolate all NAs in dataset
#'   before returning.
#'
#' @return Returns tibble expanded to have either explicit missing, or linearly
#'   interpolated values for every date from the minimum to maximum dates in the
#'   original dataset.
#'
#' @export
#'
data_to_daily <- function(data_notdaily, valuefield, interpolate = TRUE){
  #converts any interval data to daily (with NA missing) & optional linear interpolation of NA
  #obs_date field as.Date()

  #dplyr programming, NSE
  quo_valuefield <- rlang::enquo(valuefield)

  data_1day <- data_notdaily %>%
    #should handle all grouping/categories
    dplyr::group_by_at(dplyr::vars(-.data$obs_date, -!!quo_valuefield)) %>%
    #all explicit missing data - line for every Date for all groupings (above)
    tidyr::complete(obs_date = tidyr::full_seq(c(min(data_notdaily$obs_date),
                                                 max(data_notdaily$obs_date)), 1)) %>%
    dplyr::ungroup()

  if (interpolate){
    data_1day <- data_1day %>%
      #should handle all grouping/categories
      #Likely does not, actually, need to add ... for grouping variables
      dplyr::group_by_at(dplyr::vars(-.data$obs_date, -!!quo_valuefield)) %>%
      #confirm sorting
      #check if group_vars will handle grouping sorting issue
      dplyr::arrange(dplyr::group_vars(), .data$obs_date) %>%
      #will not extrapolate beyond last known value, that will happen inside run_epidemia()
      #dplyr::mutate(!!rlang::as_name(quo_valuefield) := epidemiar::na_approx(!!quo_valuefield)) %>%
      dplyr::mutate(!!rlang::as_name(quo_valuefield) := zoo::na.approx(!!quo_valuefield,
                                                                        rule = 2:1, na.rm = FALSE)) %>%
      #finish by ungrouping
      dplyr::ungroup()
  }

  data_1day
}
