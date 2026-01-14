## Standardised data format
Part of the standardised interface is to rely on a standardised data format (for the "historic data" and "weather forecast" data in the figure above).
This is a simple csv format. An example is provided in the [minimalist_example respository](https://github.com/dhis2-chap/minimalist_example). 

### Specific named fields

### Requirement for Periods to be consecutive 

### Monthly data example
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-01,10,30,200,loc1
2023-02,2,30,100,loc1
```

### Weekly data example
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-W01,12,28,45,loc1
2023-W02,8,29,52,loc1
```

The `time_period` column uses:
- `YYYY-MM` format for monthly data (e.g., `2023-01`)
- `YYYY-Wnn` format for weekly data (e.g., `2023-W01`)
