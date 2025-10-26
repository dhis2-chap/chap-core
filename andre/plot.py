import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# -----------------------------------
# 1. Load and prepare actual (gold) data
# -----------------------------------
# Read JSON file containing actual dengue case counts
with open('response_actual.json') as f:
    actual_raw = json.load(f)

# Convert the nested 'data' list into a pandas DataFrame
df_gold = pd.DataFrame(actual_raw['data'])

# Rename columns to more meaningful names
# 'pe' -> 'period' (period code as YYYYMM string)
# 'ou' -> 'org_unit' (geographic unit)
# 'value' -> 'actual' (actual case count)
df_gold = df_gold.rename(
    columns={
        'pe': 'period',
        'ou': 'org_unit',
        'value': 'actual'
    }
)

# Parse the YYYYMM string into a datetime object for time series operations
df_gold['period_dt'] = pd.to_datetime(df_gold['period'], format='%Y%m')

# -----------------------------------
# 2. Aggregate the actuals and filter to the last 10 years
# -----------------------------------
# Sum actual cases by the period date
df_gold_agg = (
    df_gold
    .groupby('period_dt')['actual']
    .sum()
    .reset_index()
)

# Determine the most recent date in the actuals
last_gold_dt = df_gold_agg['period_dt'].max()

# Select only data from the last 10 years
gold_10yr = df_gold_agg[
    df_gold_agg['period_dt'] >= last_gold_dt - DateOffset(years=10)
]

# -----------------------------------
# 3. Load and prepare forecast data
# -----------------------------------
# Read JSON file containing forecasted values
with open('response.json') as f:
    pred_raw = json.load(f)

# Build a flat list of forecast records for 1-month ahead predictions
records = []
for fx in pred_raw['forecasts']:
    # The 'period' field in the forecast is the origin month (YYYYMM)
    origin_dt = pd.to_datetime(fx['period'], format='%Y%m')
    
    # Only extract the 1-month ahead forecast (step == 1)
    for step, val in enumerate(fx['values'], start=1):
        if step == 1:
            # Compute the forecast target date by adding 1 month to origin
            target_dt = origin_dt + DateOffset(months=1)
            records.append({
                'target_dt': target_dt,
                'predicted': val
            })

# Convert the list of records into a DataFrame
df_pred = pd.DataFrame(records)

# -----------------------------------
# 4. Aggregate the forecasts and filter to the last year
# -----------------------------------
# Sum predicted cases by the target date
df_pred_agg = (
    df_pred
    .groupby('target_dt')['predicted']
    .sum()
    .reset_index()
)

# Determine the most recent forecast date
last_pred_dt = df_pred_agg['target_dt'].max()

# Select only forecasts from the last 12 months
pred_1yr = df_pred_agg[
    df_pred_agg['target_dt'] >= last_pred_dt - DateOffset(years=1)
]

# -----------------------------------
# 5. Plot actuals vs forecasts
# -----------------------------------
# Create a Matplotlib figure and axes
fig, ax = plt.subplots()

# Plot 10-year history of actual dengue cases
ax.plot(
    gold_10yr['period_dt'],
    gold_10yr['actual'],
    label='Actual (last 10 years)'
)

# Plot 12-month forecast (1-month ahead) on the same axes
ax.plot(
    pred_1yr['target_dt'],
    pred_1yr['predicted'],
    label='1-month ahead Forecast (last year)'
)

# Label axes and add a title
ax.set_xlabel('Date')
ax.set_ylabel('Dengue Cases')
ax.set_title('Dengue Cases: Actual (10-year history) vs Forecast (12 months)')

# Show a legend to differentiate the two series
ax.legend()

# Improve layout and render the plot
plt.tight_layout()
plt.show()
