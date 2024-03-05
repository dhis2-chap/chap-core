Two main paths.
One is to help develop and evaluate probabalistic forecasting models. 
The other is to run and evaluate existing methods

# Plot data with dates on the axis ..plotting
# Jointly plot monthly and daily data ..plotting
* Maybe plotly has someting for this?
# Set up github actions for weekly reports ..build
# Evaluation plots for sampler-module ..plotting/assesment
Improve *prediction_plot.forecast_plot*
- A forecast-sampler can forecast ahead for a given time ahead stochastically. 
# MultiPeriod Evaluator ..assessment
Two different modes. For sampling models: Asses for missing years/months and return forecasting plots
For lead-time specific models, predict ahead the given lead time.
# Implement naive AR/seasonal model for comparison
Include season as factor, and previous disease_count as predictors
Can also develop a similar naive sampler
# Go from health data file and method to evaluation html ..integration
# GeoCoder wrapper ..data_wrangling
# Datastructure to store data over several locations and time scales ..datatypes
There is a large logical difference between two neighbouring time points in  the same location 
and two different time points in two different locations. This is not represented in a normal (tidy) dataframe. 
Should have an interface for data where different locations are handled in a logical way. Maybe __getitem__ on location. 
# Multilocation training interface ..predictor
We need to train on data from different locations
# Start dashboard ..plotting
An evaluation of a model (or more) should give some userfriendly output. Html? Pdf? Markdown?
# Data loader for standardized data ..io
For data that is in our standardized format, we should be able to load it a bit independent of file io.
Maybe request and cache public data
# Dependency handler
We will have to run R scripts with dependencies, and generic machine learning methods with dependencies that might crash
Might need to be handled separately. hydromet_dengue/output/preds/0x_leave_month_out.R
# Running R scripts that follow an interface.
Give training data as csv file and future weather as csv file and receive csv file with predictions back. Wrapping of this should maybe be done in python 
# Caching of simulated data and downloaded data and processed data


# Meeting Agenda
* Going through acceptance tests
* Discussing interfaces:
- Datastructure to store data over several locations and time scales
- Multilocation training interface
- Report output
* Time commitment and worksession scheduling
* Assignment of tasks
* Rules for code review:
- What is important for good code for us
* Work



