# Temporary model explainability and plots

Note that this is a temporary solution and that this functionality will be integrated and expanded upon in the future, but it serves as a simple ad-hoc solution for now.

Understanding why a model makes certain predictions is important both for building trust in forecasts and for guiding model development. This page describes how to extract and visualise explanatory information from your model within the Chap framework. This tutorial will explain how to alter your model code to save the desired files generated in the model code and where to find them after a run is complete. This tutorial assumes you are using `chap evaluate` from the CLI.

We provide examples in both *R* and *Python*.

## Feature importance and coefficient size

### R

```r
df <- read.csv(future_fn)
unique_file_name <- df[1, "time_period"]

fe_df <- data.frame(
    term = rownames(model$summary.fixed),
    mean = model$summary.fixed$mean,
    sd = model$summary.fixed$sd,
    row.names = NULL
)

filepath <- paste0("explainability_", unique_file_name, ".csv")
write.csv(fe_df, filepath, row.names = FALSE)
```

The first two lines retrieve the first future time period, for example `2025-01` for January 2025. This is done at the start of the model code as the future data is sometimes merged with the historic data depending on the model used. The rest of the code chunk is after the model has been trained. The R example is based on an INLA model, where the mean and standard deviation is retrieved for each of the linear covariates. The Python example assumes a scikit-learn style model where coefficients are available via `model.coef_`. The dataframe is then written to a csv file using the `unique_file_name` to create different file names so they do not overwrite each other. Since the INLA example has all code in the `predict.R` file, the coefficients are saved there, while for other models it may make more sense to save them in the training script instead, using the last time period in the training data to generate unique filenames. The current R example can be seen in `predict.R` [here](https://github.com/chap-models/ewars_template/tree/Interpretability-for-simulation).

### Python

```python
future_df = pd.read_csv(future_fn)
unique_file_name = future_df["time_period"].iloc[0]

# After training, extract coefficients from the model.
# This example assumes a scikit-learn linear model (e.g. LinearRegression, Ridge).
fe_df = pd.DataFrame({
    "term": feature_names,
    "coefficient": model.coef_,
})

filepath = f"explainability_{unique_file_name}.csv"
fe_df.to_csv(filepath, index=False)
```

## Plotting feature importance

### R

```r
library(ggplot2)

fe_df <- read.csv(paste0("explainability_", unique_file_name, ".csv"))

p <- ggplot(fe_df, aes(x = reorder(term, mean), y = mean)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
  coord_flip() +
  labs(x = "Term", y = "Coefficient (mean +/- sd)", title = "Feature importance") +
  theme_minimal()

ggsave(paste0("feature_importance_", unique_file_name, ".png"), p, width = 8, height = 5)
```

### Python

```python
import pandas as pd
import matplotlib.pyplot as plt

fe_df = pd.read_csv(f"explainability_{unique_file_name}.csv")

fig, ax = plt.subplots()
ax.barh(fe_df["term"], fe_df["coefficient"])
ax.set_xlabel("Coefficient")
ax.set_title("Feature importance")
fig.tight_layout()
fig.savefig(f"feature_importance_{unique_file_name}.png", dpi=150)
plt.close(fig)
```

Note that the R version includes error bars based on the standard deviation from the INLA model output, while the Python version plots point estimates from a linear model. Adapt these to match the output of your specific model.


## Diagnostic plots

Diagnostic plots help verify model assumptions and identify potential issues such as patterns in residuals or poor fit for specific locations.

### R

```r
library(ggplot2)

train_df <- read.csv(train_fn)
train_df$residual <- train_df$disease_cases - train_df$predicted

p1 <- ggplot(train_df, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Predicted", y = "Residual", title = "Residuals vs Predicted") +
  theme_minimal()

ggsave(paste0("diagnostics_scatter_", unique_file_name, ".png"), p1, width = 8, height = 5)

p2 <- ggplot(train_df, aes(x = residual)) +
  geom_histogram(bins = 30) +
  labs(x = "Residual", title = "Residual distribution") +
  theme_minimal()

ggsave(paste0("diagnostics_hist_", unique_file_name, ".png"), p2, width = 8, height = 5)
```

### Python

```python
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv(train_fn)
train_df["residual"] = train_df["disease_cases"] - train_df["predicted"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(train_df["predicted"], train_df["residual"], alpha=0.5)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(train_df["residual"], bins=30)
axes[1].set_xlabel("Residual")
axes[1].set_title("Residual distribution")

fig.tight_layout()
fig.savefig(f"diagnostics_{unique_file_name}.png", dpi=150)
plt.close(fig)
```

## Finding the saved files

The saved files can be located inside your chap-core folder. Generally, the location will be `chap-core/runs/your_model/time_ran/`, together with the general files for the model. Assuming this is the folder structure you can also manually save the files to either subfolders inside the current folder or further up in the folder hierarchy if desired.