# Load required libraries for time series analysis and visualization
library(fpp3)        # Forecasting and time series functions
library(tidyverse)   # Data wrangling and visualization
library(ggdist)      # For visualizing distributions

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

# Extract diabetes drug costs (ATC2 code "A10") from PBS dataset
# Aggregate costs by month and create a tsibble (time series tibble) to calculate total cost across the country
PBS |>
  filter(ATC2 == "A10") |>
  index_by(Month) |> 
  summarise(Cost = sum(Cost)) -> cost_diabetes_drugs

# ==============================================================================
# TIME SERIES CROSS VALIDATION SETUP
# ==============================================================================

forecast_horizon <- 6  # Number of periods to forecast ahead (e.g., 6 months)
# Define the test set size as a percentage of total observations
percentage_test <- 0.3

# Split data into test set (last 30% of observations)
# Uses filter_index to select data from a calculated start date to end
test <- cost_diabetes_drugs |> 
  filter_index(
    as.character(
      max(cost_diabetes_drugs$Month) - 
        round(percentage_test * length(unique(cost_diabetes_drugs$Month))) + 1
    ) ~ .
  )

# Split data into training set (first 70% of observations)
# Uses filter_index to select data from beginning to calculated end date
train <- cost_diabetes_drugs |> 
  filter_index(
    . ~ as.character(
      max(cost_diabetes_drugs$Month) - 
        round(percentage_test * length(unique(cost_diabetes_drugs$Month)))
    )
  )

# Create rolling origin time series cross-validation sets
# .init: initial training set size (70% of data)
# .step: shift the origin forward by 1 observation at each iteration
tscv_drug <- cost_diabetes_drugs |>
  filter_index(. ~ as.character(max(cost_diabetes_drugs$Month) - forecast_horizon)) |>
  stretch_tsibble(.init = length(unique(train$Month)), .step = 1)

# ==============================================================================
# MODEL FITTING ON TIME SERIES CROSS-VALIDATION DATA
# ==============================================================================

# Fit three different time series models on the cross-validation folds:
# - snaive: Seasonal naive (uses last value from same season)
# - arima: ARIMA (AutoRegressive Integrated Moving Average)
# - ets: ETS (Error-Trend-Seasonality exponential smoothing)
# Additionally, create a combined model as simple average of ARIMA and ETS
fit <- tscv_drug |> 
  model(
    snaive = SNAIVE(Cost),
    arima = ARIMA(Cost),
    ets = ETS(Cost)
  ) |> 
  mutate(comb = (arima + ets) / 2)  # Simple ensemble: average of ARIMA and ETS

# ==============================================================================
# FORECAST GENERATION
# ==============================================================================

# Generate forecasts from all fitted models for forecast_horizon steps ahead
fcst <- fit |> forecast(h = forecast_horizon)

# ==============================================================================
# OVERALL FORECAST ACCURACY EVALUATION
# ==============================================================================

# Calculate accuracy metrics across multiple categories:
# - Point accuracy measures (ME, RMSE, MAE, etc.)
# - Interval accuracy measures (coverage, width)
# - Distribution accuracy measures (CRPS, quantile score, etc.)
fcst_accuracy <- fcst |> 
  accuracy(
    cost_diabetes_drugs,
    measures = list(
      point_accuracy_measures,
      interval_accuracy_measures,
      distribution_accuracy_measures
    )
  )

# Display selected accuracy metrics for model comparison
fcst_accuracy |> 
  select(.model, ME, RMSE, MAE, winkler, pinball, CRPS)

# Calculate Winkler score (prediction interval score) at 95% level
fcst |> 
  accuracy(
    cost_diabetes_drugs, 
    list(winkler = winkler_score), 
    level = .95
  )

# Calculate quantile score at 95th percentile
fcst |> 
  accuracy(
    cost_diabetes_drugs, 
    list(qs = quantile_score), 
    probs = .95
  )

# ==============================================================================
# ACCURACY ANALYSIS BY ROLLING ORIGIN
# ==============================================================================

# Calculate accuracy metrics for each rolling origin (.id) and model
# This shows how model performance varies across different training periods
accuracy_by_id <- fcst |> 
  accuracy(
    cost_diabetes_drugs,
    measures = list(
      point_accuracy_measures,
      interval_accuracy_measures,
      distribution_accuracy_measures
    ),
    by = c(".model", ".id")
  )

# Visualize RMSE variation across rolling origins for each model
# Boxplot shows distribution of RMSE; jitter shows individual observations
ggplot(
  data = accuracy_by_id, 
  mapping = aes(x = RMSE, y = fct_reorder(.model, RMSE))
) +
  geom_boxplot() +
  geom_jitter(width = 0.01, alpha = 0.5, height = .1) +
  ggthemes::theme_few()

# ==============================================================================
# ACCURACY ANALYSIS BY FORECAST HORIZON
# ==============================================================================

# Add forecast horizon column (h) to forecast data
# h represents how many steps ahead each forecast was made
fc_h <- fcst |>
  group_by(.id, .model) |>
  mutate(h = row_number()) |> 
  ungroup() |>
  as_fable(response = "Cost", distribution = "Cost")

# Calculate accuracy metrics stratified by forecast horizon and model
# This reveals if models perform better for shorter or longer-ahead forecasts
fc_accuracy_h <- fc_h |>
  accuracy(
    cost_diabetes_drugs,
    measures = list(
      point_accuracy_measures,
      interval_accuracy_measures,
      distribution_accuracy_measures
    ),
    by = c(".model", "h")
  )

# Visualize how forecast accuracy changes across the forecast horizon
# Shows whether error increases/decreases further into the future
ggplot(
  data = fc_accuracy_h,
  mapping = aes(x = h, y = RMSE, color = .model)
) +
  geom_point() +
  geom_line() +
  ggthemes::scale_color_colorblind() +
  ggthemes::theme_clean()

# ==============================================================================
# FORECAST VISUALIZATION
# ==============================================================================

# Fit a single seasonal naive model to the diabetes drug cost data
fit <- cost_diabetes_drugs |> 
  model(snaive = SNAIVE(Cost))

# Generate forecasts for the next 6 periods
fcst <- fit |> forecast(h = 6)

# Create distribution visualization combining forecast uncertainty, fitted values, and actual data
# Note: This approach works best for a single or small number of time series
# Note: I visulsie th elast aprt of hsitorical data , not all, otherwise foecast a are not visible
ggplot(data = fcst, aes(x = Month, ydist = Cost)) +
  ggdist::stat_halfeye(alpha = .4) +  # Show forecast distribution
  geom_line(aes(y = .mean, colour = "Point Forecast")) +  # Point forecasts
  geom_line(aes(y = .fitted, colour = "Fitted"), data = augment(fit) |> filter_index("2005 Jan" ~ .)) +  # Fitted values
  geom_point(aes(y = .fitted, colour = "Fitted"), data = augment(fit) |> filter_index("2005 Jan" ~ .)) +
  geom_line(aes(y = Cost, colour = "Data"), data = cost_diabetes_drugs |> filter_index("2005 Jan" ~ .)) +  # Actual data
  geom_point(aes(y = Cost, colour = "Data"), data = cost_diabetes_drugs |> filter_index("2005 Jan" ~ .)) +
  scale_color_manual(
    name = NULL,
    breaks = c("Fitted", "Data", "Point Forecast"),
    values = c("Fitted" = "#E69F00", "Data" = "#0072B2", "Point Forecast" = "#000000")
  )
