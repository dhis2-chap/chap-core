# Data requirements in CHAP

CHAP expects the data to contain certain features as column names of the supplied csv files. Specifically, time_period, population, disease_cases, location, rainfall and mean_temperature. CHAP gives an error if any of these are missing in the supplied datafile. Additionally, there are convntions for how to represent time in the time_period. For instance, weekly data should be represented as 
|time_period|
|-----------------|
|2014-12-29/2015-01-04|
|2015-01-05/2015-01-11|
|2015-01-12/2015-01-18|
|2015-01-19/2015-01-25|

And for monthly data it should be

|time_period|
|-----------------|
|2014-12|
|2015-01|
|2015-02|
|2015-03|

This requirement is checked for supplied data files for training and predicitng and also for the output data from the model. Additionaly the model should give samples from a distribuition, preferable $1000$ samples for each location and time index with column names `sample_0, sample_1` and so on.

A useful tool for handling data is the adapters that can be included in the MLproject file. These adapters map the internal names in CHAP to whatever you want them to be. For instance disease_cases could be mapped to cases, as in the MLproject file in the repository under the [dhis2-chap organization](https://github.com/dhis2-chap/chap_auto_ewars_weekly). In practice, the adapters copy the mentioned column and gives it the new column name.