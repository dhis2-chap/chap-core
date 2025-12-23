# Making chap-compatible train and predict endpoints

To integrate a component for standardized, interoperable use, it must follow an established standard. CHAP defines one such standard, and by adhering to it, your code gains all the benefits of seamless platform integration. In predictive modeling and machine learning, it is a long-established best practice to provide separate functions for training and prediction.
 
The following figure shows the basic overview of how CHAP expects modelling code to be, i.e. divided into separated _train_ and _predict_ parts

![External model structure](../_static/modelling_code.png)

The figure below shows how the chap platform orchestrates training and prediction of your model using the same endpoints as above:

![External model interaction](../_static/modelling_code_with_chap.png)

The exact way of specifying the train and predict endpoints are described [here](describe_model.md).

## Standardised data format
Part of the standardised interface is to rely on a standardised data format (for the "historic data" and "weather forecast" data in the figure above).
This is a simple csv format. An example is provided in the [minimalist_example respository](https://github.com/dhis2-chap/minimalist_example).

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

