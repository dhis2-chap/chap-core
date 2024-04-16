#Data Cleaning and Helper function

## Data prep & cleaning functions

#' Interpolates missing epi data.
#'
#' @param quo_casefield Quosure of user given casefield to run_epidemia().
#' @param quo_groupfield Quosure of the user given geographic grouping field to
#'   run_epidemia().
#'
#'@inheritParams run_epidemia
#'
#' @return Same data as epi_data with new interpolated case field,
#'   cases_epidemiar.
#'
#'
epi_NA_interpolate <- function(epi_data, quo_casefield, quo_groupfield){
  epi_data %>%
    dplyr::group_by(!!quo_groupfield) %>%
    #confirm geogroup-date sorting
    dplyr::arrange(!!quo_groupfield, .data$obs_date) %>%
    #interpolate, but not on trailing edge
    #dplyr::mutate(cases_epidemiar = epidemiar::na_approx(!!quo_casefield)) %>%
    dplyr::mutate(cases_epidemiar = zoo::na.approx(!!quo_casefield, rule=2:1, na.rm = FALSE)) %>%
    #force into integer after interpolating (could cause problems with modeling otherwise)
    dplyr::mutate(cases_epidemiar = floor(.data$cases_epidemiar)) %>%
    #finish by ungrouping
    dplyr::ungroup()
}


#' #' Interpolates missing environmental data.
#' #' Deprecated, no longer used as extend_env_data() will fill any gaps.
#' #'
#' #' @param quo_obsfield Quosure of the user given field that holds the
#' #'   environmental variable identifiers/names/IDs.
#' #' @param quo_valuefield Quosure of the user given field that holds the
#' #'   environmental variable observation value.
#' #' @param quo_groupfield Quosure of the user given geographic grouping field to
#' #'   run_epidemia().
#' #'
#' #'@inheritParams run_epidemia
#' #'
#' #' @return Same data as env_data, with new interpolated field, val_epidemiar, of
#' #'   the environmental variable data.
#' #'
#' env_NA_interpolate <- function(env_data, quo_obsfield, quo_valuefield, quo_groupfield){
#'   env_data %>%
#'     #first, mark which ones are observed versus (will be) interpolated
#'     dplyr::mutate(data_source = ifelse(!is.na(!!quo_valuefield), "Observed", "Interpolated")) %>%
#'     #two levels of group_by
#'     dplyr::group_by(!!quo_groupfield, !!quo_obsfield) %>%
#'     #confirm date sorting
#'     dplyr::arrange(!!quo_groupfield, !!quo_obsfield, .data$obs_date) %>%
#'     #interpolate
#'     #dplyr::mutate(val_epidemiar = !!quo_valuefield,
#'     #              val_epidemiar = epidemiar::na_approx(.data$val_epidemiar)) %>%
#'     dplyr::mutate(val_epidemiar = zoo::na.approx(!!quo_valuefield, rule = 2:1, na.rm = FALSE)) %>%
#'     #finish by ungrouping
#'     dplyr::ungroup()
#' }



#Helper functions
#' Create a named list.
#'
#' Creates a named list from the user given items. Will preserve the names of
#' items that already have names.
#'
#' @param ... List of objects, named or not, to be included in the fully named
#'   list.
#'
#' @examples
#' a <- list("a", "aa", "aaa")
#' b <- data.frame(x = 1:4, y = 5:8)
#' create_named_list(a, b, c = rep(1:4))
#'
#' @export
#'
create_named_list <- function(...){
  list_to_name <- list(...)
  named_chr <- sapply(substitute(list(...)),deparse)[-1]
  given_names <- names(list_to_name)
  #if given no names, then just use names of original items
  if (is.null(given_names)) {
    names_to_use <- named_chr
  }
  #logical vector of which items did not have names given/to be assigned
  not_named_logical <- given_names == ""
  if (any(not_named_logical)) {
    #set up, names that were given
    names_to_use <- given_names
    #add names for things that had names originally
    names_to_use[not_named_logical] <- named_chr[not_named_logical]
  }
  #set the names
  stats::setNames(list_to_name, names_to_use)
}

#' Mode
#'
#' Calculate the mode of a set of values, for numeric or character/factor data.
#' In ties, returns the first tied value.
#'
#' @param x A vector.
#' @param na.rm Logical indicating whether \code{NA} values should be excluded.
#'
#' @return A vector of length 1 and the same class as \code{x}.
#' @export
#'
#' @examples
#' Mode(c(1,1,2,3))
#' Mode(c(1,2,2,3))
#' Mode(c(1,1,3,3))
#' Mode(c(3,3,1,1))
#' Mode(c(1,NA,NA))
#' Mode(c(1,NA,NA), na.rm = TRUE)
#'
Mode <- function(x, na.rm = FALSE) {
  if (!is.vector(x)) stop("x is not a vector, but is class ",
                          paste(class(x), collapse = ", "))
  ux <- unique(x)
  if (na.rm) ux <- stats::na.exclude(ux)
  ux[which.max(tabulate(match(x, ux)))]
}

