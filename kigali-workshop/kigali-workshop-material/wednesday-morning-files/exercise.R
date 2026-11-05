############################################################
# Modelling forecast uncertainty (Workshop Exercises)
# Harsha Halgamuwe Hewage | DL4SG, Cardiff University
# Date: 2026-02-25
#
# Goal:
#   1) Model-based uncertainty (ARIMAX distributions + prediction intervals)
#   2) Bootstrap uncertainty (sample paths + bootstrap intervals)
#   3) Conformal prediction (split/cv style calibration -> valid PI)
#
# How to use this script:
#   - Run section-by-section (Ctrl/Cmd + Enter).
#   - Do NOT rush to the end. Each "YOUR TURN" is an exercise.
############################################################


############################
# 0) Housekeeping
############################

# Clear workspace (optional)
rm(list = ls())

# If you want reproducibility for bootstrap simulations:
set.seed(123)

# ---- Packages: install if missing, then load ----
required_packages <- c(
  "tidyverse",
  "ggthemes",
  "tsibble",
  "feasts",
  "fable",
  "distributional",
  "conformalForecast",
  "forecast",
  "ggdist"        # used for half-eye distribution plot
)

new_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if (length(new_packages)) {
  install.packages(new_packages, dependencies = TRUE)
}

invisible(lapply(required_packages, library, character.only = TRUE))


############################
# 1) Load data + create tsibble
############################

# ---- IMPORTANT ----
# This script expects the dataset:
#   data/brazil_dengue.csv
#
# If your file is elsewhere, change the path below.

brazil_dengue <- read_csv('data/brazil_dengue.csv', show_col_types = FALSE)

# Create a tsibble:
# - index = time_period (monthly)
# - key   = location (multiple locations)
brazil_dengue_tsb <- brazil_dengue |>
  mutate(time_period = yearmonth(time_period)) |>
  as_tsibble(index = time_period, key = location)

# Quick check
brazil_dengue_tsb |> head(5)
brazil_dengue_tsb |> count(location) |> print()

# ---- Plot one series (Bahia) ----
brazil_dengue_tsb |>
  filter(location == "Bahia") |>
  autoplot(disease_cases) +
  labs(x = "Month", y = "Disease Cases") +
  theme_minimal(base_family = "Inter", base_size = 14) +
  theme(panel.border = element_rect(color = "lightgrey", fill = NA))

# =========================
# YOUR TURN (Exercise 1)
# =========================
# 1) Change "Bahia" to another location you see in the data.
# 2) Plot disease_cases again.
# 3) Describe in 1 sentence: Is it seasonal? Are there spikes?


############################
# 2) Train/Future split (simple)
############################

# Here we use:
#   Train: up to 2015 Dec
#   Future: from 2016 Jan onwards (exogenous variables available, disease_cases removed)
brazil_dengue_tsb_train <- brazil_dengue_tsb |>
  filter_index(. ~ "2015 Dec")

brazil_dengue_tsb_future <- brazil_dengue_tsb |>
  filter_index("2016 Jan" ~ .) |>
  select(-disease_cases)

# Check
range(brazil_dengue_tsb_train$time_period)
range(brazil_dengue_tsb_future$time_period)


############################################################
# 3) MODEL-BASED UNCERTAINTY (ARIMAX via fable)
############################################################

# Fit ARIMAX for one location (Bahia)
fit_arimax <- brazil_dengue_tsb_train |>
  filter(location == "Bahia") |>
  model(arimax = ARIMA(disease_cases ~ rainfall + mean_temperature))

report(fit_arimax)

# Forecast distribution for the future period
fc_arimax <- fit_arimax |>
  forecast(new_data = brazil_dengue_tsb_future)

# ---- Visualise forecast distribution (half-eye) ----
# We also truncate distributions at 0 so we don't show negative cases.
fcst <- fc_arimax |> 
  mutate(disease_cases = distributional::dist_truncated(disease_cases, lower = 0), .mean=mean(disease_cases))

fitted_arimax <- fit_arimax |> augment()

ggplot(data = fcst, mapping = aes(x = time_period, ydist = disease_cases))+
  ggdist::stat_halfeye(alpha = .4) +
  geom_line(aes(y=.mean, colour ="Point Forecast")) +
  geom_line(aes(y = .fitted, colour ="Fitted"), data = filter_index(fitted_arimax, "2015 Jan" ~ .)) +
  geom_point(aes(y = .fitted, colour ="Fitted"), data = filter_index(fitted_arimax, "2015 Jan" ~ .)) +
  geom_line(aes(y = disease_cases, colour ="Actual"), data = filter_index(brazil_dengue_tsb |> filter(location == 'Bahia'), "2016 Jan" ~ .)) +
  geom_point(aes(y = disease_cases, colour ="Actual"), data = filter_index(brazil_dengue_tsb |> filter(location == 'Bahia'), "2016 Jan" ~ .))+
  scale_color_manual(name=NULL,
                     breaks=c('Fitted', 'Actual',"Point Forecast"),
                     values=c('Fitted'='#E69F00', 'Actual'='#0072B2',"Point Forecast"="#000000"))+
  theme_minimal() +
  theme(panel.border = element_rect(color = "lightgrey", fill = NA))

# =========================
# YOUR TURN (Exercise 2)
# =========================
# 1) Fit ARIMAX for a different location.
#    Hint: replace filter(location == "Bahia") with your chosen location.
# 2) Re-run the forecast distribution plot.
# 3) Does truncation at 0 change how you interpret uncertainty?


############################
# 4) Prediction intervals (model-based)
############################

# Extract intervals using hilo()
pi_tbl <- fit_arimax |>
  forecast(new_data = brazil_dengue_tsb_future) |>
  hilo(level = c(80, 95)) |>
  select(location, .model, .mean, `80%`, `95%`)

pi_tbl |> head(6)

# Unpack 90% PI as a quick example
pi_90 <- fit_arimax |>
  forecast(new_data = brazil_dengue_tsb_future) |>
  hilo(level = 90) |>
  select(location, .model, .mean, `90%`) |>
  unpack_hilo(`90%`)

pi_90 |> head(6)

# Plot intervals with autoplot
fit_arimax |>
  forecast(new_data = brazil_dengue_tsb_future) |>
  autoplot(
    brazil_dengue_tsb |>
      filter(location == "Bahia") |>
      filter_index("2015 Jan" ~ .)
  ) +
  labs(x = "Month", y = "Disease Cases") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "lightgrey", fill = NA))

# =========================
# YOUR TURN (Exercise 3)
# =========================
# 1) Change PI level from 95% to 90% and 99%.
# 2) Plot them.
# 3) Write one sentence: what happens as level increases?


############################################################
# 5) BOOTSTRAP UNCERTAINTY (sample paths + bootstrap intervals)
############################################################

# Bootstrap sample paths using generate()
sim <- fit_arimax |>
  generate(
    new_data = brazil_dengue_tsb_future,
    bootstrap = TRUE,
    times = 5      # increase to 100 for smoother-looking plots
  )

# Plot actual history + simulated futures
brazil_dengue_tsb |> 
  filter_index('2015 Jan' ~ .) |> 
  filter(location == 'Bahia') |>
  ggplot(aes(x = time_period)) +
  geom_line(aes(y = disease_cases)) +
  geom_line(aes(y = .sim, colour = as.factor(.rep)),
            data = sim)+
  labs(y = "Disease Cases", x = "Month", colour="Future") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "lightgrey", fill = NA))

# Bootstrap-based prediction intervals
boot_pi <- fit_arimax |>
  forecast(
    new_data = brazil_dengue_tsb_future,
    bootstrap = TRUE,
    times = 1000   # more times = more stable empirical intervals
  ) |>
  hilo(level = c(80, 95))

boot_pi |> head(6)

# Plot bootstrap intervals
fit_arimax |> 
  forecast(new_data = brazil_dengue_tsb_future, bootstrap = TRUE, times = 100) |> 
  autoplot(brazil_dengue_tsb |> 
             filter(location == 'Bahia') |>
             filter_index('2015 Jan' ~ .)) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "lightgrey", fill = NA))

# =========================
# YOUR TURN (Exercise 4)
# =========================
# 1) Change times = 50, 200, 1000 and compare the PI stability.
# 2) Try another location.
# 3) Does bootstrap give wider or narrower intervals than model-based?


############################################################
# 6) CONFORMAL PREDICTION (using conformalForecast + forecast::Arima)
############################################################

# We will:
#  - pick one location (Bahia)
#  - create train/calibration/test splits by dates
#  - compute calibration errors using cvforecast (1-step ahead)
#  - build a conformal PI: yhat +/- q_alpha where q_alpha is a quantile of abs errors

bahia <- brazil_dengue_tsb |>
  filter(location == "Bahia") |>
  arrange(time_period) |>
  as_tibble()

# Target series
y_all <- ts(bahia$disease_cases, frequency = 12)

# Exogenous regressors
xreg_all <- bahia |>
  select(rainfall, mean_temperature) |>
  as.matrix()

# Define split dates (same spirit as your slides)
train_end   <- yearmonth("2014 Dec")
calib_start <- yearmonth("2015 Jan")
calib_end   <- yearmonth("2015 Dec")
test_start  <- yearmonth("2016 Jan")
test_end    <- yearmonth("2016 Dec")

idx <- bahia$time_period

i_train_end  <- max(which(idx <= train_end))
i_calib_end  <- max(which(idx <= calib_end))
i_test_start <- min(which(idx >= test_start))
i_test_end   <- max(which(idx <= test_end))

# Window length used for rolling CV inside cvforecast
window_len <- i_train_end

# We build y up to calibration end (train + calib),
# because cvforecast will generate 1-step errors across the calibration months.
y_train_calib <- ts(bahia$disease_cases[1:i_calib_end], frequency = 12)
xreg_train_calib <- xreg_all[1:i_calib_end, , drop = FALSE]

# ---- Forecast function for cvforecast (ARIMAX) ----
# conformalForecast expects a function:
#   forecastfun(x, h, level) -> forecast object (forecast::forecast)
farimax_1step <- function(x, h, level) {
  t <- length(x)
  
  # Match xreg rows to the training window length t
  xreg_train  <- xreg_train_calib[1:t, , drop = FALSE]
  xreg_future <- xreg_train_calib[(t + 1):(t + h), , drop = FALSE]
  
  fit <- forecast::Arima(x, xreg = xreg_train)
  forecast::forecast(fit, h = h, level = level, xreg = xreg_future)
}

# ---- Calibration residuals via rolling CV ----
# This produces 1-step ahead errors for each month in the calibration year.
fc_cal <- conformalForecast::cvforecast(
  y = y_train_calib,
  forecastfun = farimax_1step,
  h = 1,
  level = 95,
  forward = FALSE,
  window = window_len,
  initial = 1
)

# Conformity scores: absolute errors on calibration points
scores <- abs(as.numeric(fc_cal$ERROR))

# Choose alpha = 0.05 for a 95% interval
q_95 <- as.numeric(stats::quantile(scores, probs = 0.95, na.rm = TRUE))

# ---- Fit final model on train+calib, forecast test ----
fit_final <- forecast::Arima(
  ts(bahia$disease_cases[1:i_calib_end], frequency = 12),
  xreg = xreg_all[1:i_calib_end, , drop = FALSE]
)

h_test <- (i_test_end - i_test_start + 1)

fc_test <- forecast::forecast(
  fit_final,
  h = h_test,
  xreg = xreg_all[i_test_start:i_test_end, , drop = FALSE]
)

test_dates <- bahia$time_period[i_test_start:i_test_end]
yhat <- as.numeric(fc_test$mean)

plot_fc <- tibble(
  time_period = test_dates,
  point = yhat,
  lower_95 = pmax(0, yhat - q_95),
  upper_95 = yhat + q_95,
  actual = bahia$disease_cases[i_test_start:i_test_end]
)

history <- bahia |>
  filter(time_period >= yearmonth("2014 Jan") & time_period <= calib_end)

ggplot() +
  geom_line(
    data = history,
    aes(x = time_period, y = disease_cases, colour = "Actual"),
    linewidth = 1.2
  ) +
  geom_ribbon(
    data = plot_fc,
    aes(x = time_period, ymin = lower_95, ymax = upper_95, fill = "95% Conformal PI"),
    alpha = 0.2
  ) +
  geom_line(
    data = plot_fc,
    aes(x = time_period, y = point, colour = "Point Forecast"),
    linewidth = 1
  ) +
  geom_line(
    data = plot_fc,
    aes(x = time_period, y = actual, colour = "Actual"),
    linewidth = 1
  ) +
  scale_colour_manual(
    name = NULL,
    values = c(
      "Point Forecast" = "#0072B2",
      "Actual" = "black"
    )
  ) +
  scale_fill_manual(
    name = NULL,
    values = c("95% Conformal PI" = "#0072B2")
  ) +
  guides(
    colour = guide_legend(order = 1, override.aes = list(linewidth = 1)),
    fill   = guide_legend(order = 2)
  ) +
  labs(x = "Month", y = "Disease Cases") +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "lightgrey", fill = NA),
    legend.position = "right"
  )

# =========================
# YOUR TURN (Exercise 5)
# =========================
# 1) Change the split:
#    - Make calibration 24 months instead of 12 months (e.g., 2015-2016).
#    - Move the test year forward.
# 2) Recompute q_95 and replot.
# 3) Does the interval get tighter or wider? Why?
