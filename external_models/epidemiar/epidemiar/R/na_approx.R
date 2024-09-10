#' #' Replace NA by interpolation in a numeric vector
#' #'
#' #' Fill in missing values in a numeric vector by approximat them through linear
#' #' interpolation.
#' #'
#' #' @param x A numeric vector.
#' #' @param order.by An index vector with unique entries by which the observations
#' #'   in `x`` are ordered.
#' #'
#' #' @details Leading and trailing `NA`s are left as is.
#' #'
#' #' @return A numeric vector of the same length as `x`.
#' #' export
#' #'
#' #' @examples
#' #' na_approx(c(NA, .31, 4, NA, NA, 10, NA))
#' na_approx <- function(x, order.by = zoo::index(x)) {
#'   n_non_NAs <- sum(!is.na(x)) # number of non-NA values in x
#'   if(n_non_NAs < 2) return(x)
#'   zoo::zoo(x, order.by) %>%
#'     zoo::na.approx(na.rm = FALSE) %>%
#'     zoo::coredata()
#' }
