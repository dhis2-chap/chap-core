#' Adds ISO or CDC datefields to a data set with a "obs_date" (Date) field
#'
#' @param df Data table or tibble with a field named "obs_date"
#' @param type String to indicate whether to use "ISO" ISO-8601 week of year
#'   (used by WHO) or "CDC" epi weeks.
#'
#' @return dataframe (or tibble) with week (`week_epidemiar`) and year
#'   (`year_epidemiar`) fields added
#'
#' @export
#'
#'

add_datefields <- function(df, type = "ISO"){

  week_type <- match.arg(type, c("ISO", "CDC"))

  # adds year and week numbers
  if (week_type == "ISO"){
    df <- df %>%
      #add iso wk/yr
      dplyr::mutate(week_epidemiar = lubridate::isoweek(.data$obs_date),
                    year_epidemiar = lubridate::isoyear(.data$obs_date))

  } else if (week_type == "CDC"){
    #add CDC epi wk/yr
    dplyr::mutate(week_epidemiar = lubridate::epiweek(.data$obs_date),
                  year_epidemiar = lubridate::epiyear(.data$obs_date))
  }

  df

}
