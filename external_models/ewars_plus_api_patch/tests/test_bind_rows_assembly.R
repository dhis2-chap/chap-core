# Regression test for CLIM-617.
#
# The CV step in Lag_Model_selection_ewars_By_District_api.R writes one RDS
# per (year, weekly-window) fold. Lag-selection picks a different optimal
# lag per year, so each per-year frame carries differently-named lag-suffixed
# columns. The original `foreach(.combine = rbind)` assembly refused to stack
# frames with mismatched column names; the patch switches to
# `dplyr::bind_rows(lapply(...))` which fills missing columns with NA.
#
# This test exercises the assembly logic directly on synthetic per-year
# frames so it does not require a full CV run. Old path is asserted to
# fail with the original error; new path is asserted to succeed and
# produce the union-of-columns shape expected by downstream code.
#
# Run inside the patched image:
#   docker run --rm chap-core/ewars_plus_api:clim-617 \
#     Rscript /home/app/tests/test_bind_rows_assembly.R

suppressPackageStartupMessages({
  library(foreach)
  library(dplyr)
})

# ---- fixtures ------------------------------------------------------------
# Three per-year frames, each with a different optimal-lag suffix on the two
# alarm-variable columns. Common columns mirror the real CV output (subset).
make_year_frame <- function(year, lag_n) {
  weeks <- 1:4
  df <- data.frame(
    district           = 1L,
    year               = year,
    week               = weeks,
    Cases              = c(10L, 12L, 11L, 13L),
    Pop                = 1e6,
    mean               = c(11.0, 12.5, 10.8, 13.2),
    lci                = c(2,    3,    2,    3),
    uci                = c(38,   42,   35,   44),
    mean_rate          = c(1.0, 1.1, 1.0, 1.2),
    sd_rate            = c(0.4, 0.5, 0.4, 0.5),
    week_Interval      = "01_04",
    stringsAsFactors   = FALSE
  )
  # The columns that drift across years.
  df[[paste0("mean_temperature_LAG", lag_n)]] <- c(25.1, 26.0, 27.4, 28.1)
  df[[paste0("rainfall_std_LAG",     lag_n)]] <- c(-0.7, -0.3, 0.5, 1.4)
  df
}

frames <- list(
  make_year_frame(2022, 12L),
  make_year_frame(2023, 10L),
  make_year_frame(2024,  7L)
)

stopifnot(length(frames) == 3L)

# ---- 1. old path: must fail with the historical error message -------------
old_res <- tryCatch(
  foreach(i = seq_along(frames), .combine = rbind) %do% frames[[i]],
  error = function(e) e
)

if (!inherits(old_res, "error")) {
  stop("REGRESSION: old foreach(.combine = rbind) unexpectedly succeeded; ",
       "the bug we were guarding against is no longer reproducing — review ",
       "the test or upstream changes")
}
if (!grepl("names do not match", conditionMessage(old_res), fixed = TRUE)) {
  stop("Old path failed but with an unexpected message: ",
       conditionMessage(old_res))
}
cat("OK old path failed as expected: ", conditionMessage(old_res), "\n", sep = "")

# ---- 2. new path: must succeed and produce union-of-columns ---------------
new_res <- dplyr::bind_rows(frames)

stopifnot(nrow(new_res) == 12L)                                # 3 years x 4 weeks
stopifnot(all(c("district","year","week","Cases","Pop","mean",
                "lci","uci","mean_rate","sd_rate","week_Interval") %in% names(new_res)))

# All three lag-suffix variants should coexist; each is non-NA only for its
# originating year and NA elsewhere. This is the property downstream code
# relies on (it does not read the lag-suffixed columns).
for (lag_n in c(12L, 10L, 7L)) {
  for (col in c(paste0("mean_temperature_LAG", lag_n),
                paste0("rainfall_std_LAG",     lag_n))) {
    stopifnot(col %in% names(new_res))
    stopifnot(sum(!is.na(new_res[[col]])) == 4L)               # one year worth
    stopifnot(sum( is.na(new_res[[col]])) == 8L)               # the other two
  }
}
cat("OK new path produced rows=", nrow(new_res),
    " cols=", ncol(new_res), "\n", sep = "")

cat("PASS: CLIM-617 regression test\n")
