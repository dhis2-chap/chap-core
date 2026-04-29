# Regression test for CLIM-617 (predict-side: non-conformable arrays).
#
# In DBII_predictions_Vectorized_API.R the prospective threshold is built by
# looking up each prospective week's row in the precomputed endemic-channel
# table:
#
#   idx.end_all <- foreach(aa = Prospective_Data_with_inla_grp$week,
#                          .combine = c) %do% which(endemic_channel_Use$week == aa)
#
# When a prospective week is not in the endemic table, `which(...)` returns
# integer(0) and `foreach(.combine = c)` silently drops it. idx.end_all
# becomes shorter than the prospective row count; downstream the threshold
# matrix is also shorter; the elementwise comparison
#
#   prob_Exceed_mat <- ypred_NB_1000_rate_all > week_rate_threshold_all
#
# crashes with `<simpleError ... non-conformable arrays>` and the wrapper
# surfaces it as `RuntimeError: EWARS API error from /Ewars_predict: 500`.
#
# The patch replaces the foreach idiom with `match()`, which preserves
# length: unmatched weeks become NA_integer_, the threshold cell becomes
# NA_real_, and the comparison harmlessly produces NA in those rows.
#
# This test exercises that lookup logic on synthetic frames so it does not
# require a real /Ewars_predict run.
#
# Run inside the patched image:
#   docker run --rm chap-core/ewars_plus_api:clim-617 \
#     Rscript /home/app/tests/test_idx_end_match.R

suppressPackageStartupMessages({
  library(foreach)
})

# ---- fixtures ------------------------------------------------------------
# Prospective weeks include 999, which is intentionally not in the endemic
# table — that is the failure trigger.
Prospective_Data_with_inla_grp <- data.frame(
  week = c(13L, 14L, 15L, 999L, 16L)
)
endemic_channel_Use <- data.frame(
  week           = 1:52,
  threshold_rate = seq(1.0, by = 0.1, length.out = 52)
)
n_prospective <- nrow(Prospective_Data_with_inla_grp)

# Stand-in for the predicted-rate matrix; the values do not matter for the
# regression — only the shape does.
ypred_NB_1000_rate_all <- matrix(1.5,
                                 nrow = n_prospective,
                                 ncol = 1000)

# ---- 1. old path: must reproduce the historical "non-conformable arrays" ---
old_idx <- foreach(aa = Prospective_Data_with_inla_grp$week, .combine = c) %do%
             which(endemic_channel_Use$week == aa)

if (length(old_idx) == n_prospective) {
  stop("REGRESSION: the old foreach(.combine=c)+which idiom no longer drops ",
       "unmatched weeks; this test (and the CLIM-617 bug) no longer reproduce")
}
stopifnot(length(old_idx) == n_prospective - 1L)  # week=999 was dropped

old_threshold0 <- endemic_channel_Use$threshold_rate[old_idx]
old_threshold  <- replicate(1000, old_threshold0, simplify = "matrix")

old_compare <- tryCatch(
  ypred_NB_1000_rate_all > old_threshold,
  error = function(e) e
)
if (!inherits(old_compare, "error")) {
  stop("REGRESSION: old comparison unexpectedly succeeded; the bug ",
       "we were guarding against is no longer reproducing")
}
if (!grepl("non-conformable arrays", conditionMessage(old_compare),
           fixed = TRUE)) {
  stop("Old path failed but with an unexpected message: ",
       conditionMessage(old_compare))
}
cat("OK old path failed as expected: ",
    conditionMessage(old_compare), "\n", sep = "")

# ---- 2. new path: must succeed and produce NAs for the unmatched week ----
new_idx <- match(Prospective_Data_with_inla_grp$week,
                 endemic_channel_Use$week)

stopifnot(length(new_idx) == n_prospective)
stopifnot(is.na(new_idx[4]))                    # week=999
stopifnot(!any(is.na(new_idx[-4])))

new_threshold0 <- endemic_channel_Use$threshold_rate[new_idx]
stopifnot(length(new_threshold0) == n_prospective)
stopifnot(is.na(new_threshold0[4]))

new_threshold <- replicate(1000, new_threshold0, simplify = "matrix")
stopifnot(identical(dim(new_threshold), dim(ypred_NB_1000_rate_all)))

new_compare <- ypred_NB_1000_rate_all > new_threshold
stopifnot(identical(dim(new_compare), c(n_prospective, 1000L)))
stopifnot(all(is.na(new_compare[4, ])))         # week=999 row → NAs
stopifnot(!any(is.na(new_compare[-4, ])))       # other rows → real values

# Downstream uses apply(..., mean) per row; with na.rm=TRUE that should
# return NaN/NA for the unmatched row but a real value for the others —
# check that the comparison shape is consumable without further crashes.
row_means <- apply(new_compare, 1, function(x) mean(x, na.rm = TRUE))
stopifnot(length(row_means) == n_prospective)

cat("OK new path produced ", n_prospective, "x", ncol(new_compare),
    " comparison with NA at row 4 (week=999)\n", sep = "")

cat("PASS: CLIM-617 predict-side regression test\n")
