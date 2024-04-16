
#' Get epidemiological weekday number of a date-time
#'
#' Returns the epidemiological weekday number using either ISO or CDC
#' system.
#'
#' The WHO system uses the ISO 8601 standard in which weeks start on Monday,
#' while in the CDC system weeks start on Sunday.
#'
#' @inheritParams lubridate::isoweek
#' @inheritParams lubridate::epiweek
#'
#'@param system String indicating the standard (WHO ISO-8601 or CDC epi
#'  weeks) that the weeks of the year in epidemiological and environmental
#'  reference data use ["ISO" or "CDC"]. (Required: epidemiological observation
#'  dates listed are LAST day of week).#'
#' @return The weekday number (1--7) as an integer vector.
#'
#' @inherit lubridate::isoweek references
#' @inherit lubridate::epiweek references
#'
#' @examples
#' epiwday(as.Date("2005-01-01")) # 6
#' epiwday(as.Date("2005-01-01"), system = "ISO") # 6
#' epiwday(as.Date("2005-01-01"), system = "CDC") # 7
#'
#' @export
#'
epiwday <- function(x, system = "ISO") {

  week_type <- match.arg(system, c("ISO", "CDC"))

  if (week_type == "ISO") {
    as.integer(lubridate::wday(x - 1))

  } else if (week_type == "CDC") {
    as.integer(lubridate::wday(x))
  }

}
