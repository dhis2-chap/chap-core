# Learning statistical and machine learning modelling - with climate-sensitive disease forecasting as focus and case
This material will gradually introduce you to important concepts from statistical modelling and machine learning, 
focusing on what you will need to understand in order to do forecasting of climate-sensitive disease.
It thus selects a set of topics needed for disease forecasting, while mostly introducing these concepts in generality.

The material is organised around hands-on tutorials, where you will learn how to practically develop models while learning the theoretical underpinnings.

# Prerequisites

- Programming: you must know some programming, in either Python or R, to be able to follow our exercises and tutorials. 
- Data science: You should know basic data science and statistics or machine learning. However, very little is presumed as it can be learnt underway, but one should know the most basic concepts and terminology.
- GitHub: Central to our approach is that you follow various practical tutorials along the way. These tutorials are available at GitHub, so you need to know at least the absolute basics of how to get code from GitHub to your local machine - if not please ask us for tips on Github tutorials.

# Background
Climate change is rapidly reshaping the patterns and spread of disease, posing urgent challenges for health systems. To effectively respond, the health systems must become more adaptive and data-driven. Early warning and response systems (EWS) that leverage disease incidence forecasting offer a promising way to prioritize interventions where and when they are most needed.

At the core of early warning is forecasting of disease incidence forward in time. This is based on learning a statistical/machine learning model of how disease progresses ahead in time based on various available data. 

If you have limited prior experience with statistics or machine learning, please read our brief intro in the expandable box below:

??? info "A gentle introduction to statistical and time series modelling"

    ### 1. Statistical or Machine Learning Model
    A **model** is a rule or formula we create to describe how some outcome depends on information we already have.

    - The outcome we want to understand or predict is often called the **target**.
    - The information we use to make that prediction is called **predictors** (or **features**).

    A model tries to capture patterns in data in a simplified way.  
    You can think of a model as a machine:

    > **input (predictors)** → *model learns a pattern* → **output (prediction)**

    The goal is either to **explain** something (“What affects sales?”) or to **predict** something (“What will sales be tomorrow?”).

    ---

    ### 2. Predictors (Features)
    A **predictor** is any variable that provides information helpful for predicting the target.

    Examples:
    - Temperature when predicting ice cream sales  
    - Age when predicting income  
    - Yesterday’s stock price when predicting today’s  

    Predictors are the model’s *inputs*.  
    We usually write them as numbers:  
    - A single predictor as *x*  
    - Several predictors as *x₁, x₂, x₃,* …  

    The model learns how each predictor is related to the target.

    ---

    ### 3. Linear Regression
    **Linear regression** is one of the simplest and most widely used models.  
    It assumes that the target is approximately a **straight-line combination** of its predictors.

    With one predictor *x*, the model is:

    > **prediction = a + b·x**

    - **a** is the model’s baseline (what we predict when *x = 0*)  
    - **b** tells us how much the prediction changes when *x* increases by 1 unit  

    With multiple predictors *x₁, x₂, …*, we extend the same idea:

    > **prediction = a + b₁·x₁ + b₂·x₂ + …**

    You don’t need to imagine shapes in many dimensions—just think of it as a recipe where each predictor gets a weight (**b**) that shows how important it is.

    The model “learns” values of *a, b₁, b₂, …* by choosing them so that predictions are as close as possible to the observed data.

    ---

    ### 4. Time Series
    A **time series** is a sequence of data points collected over time, in order:

    > value at time 1, value at time 2, value at time 3, …

    Examples:
    - Daily temperatures  
    - Hourly website traffic  
    - Monthly number of customers  

    What makes time series special is that:

    - **The order matters**  
    - **Past values can influence future values**  
    - Data may show **patterns** such as trends (general increase/decrease over time) or seasonality (repeating patterns, like higher electricity use every winter)

    ---

    ### 5. Time Series Forecasting
    **Forecasting** means using past observations to predict future ones.

    Unlike models that treat each data point separately, forecasting models learn ideas like:

    - how the series tends to move (trend)  
    - whether it repeats patterns (seasonality)  
    - how strongly the recent past influences the next value  

    A simple forecasting idea is to predict the next value using a weighted average of recent past values. More advanced methods learn more complex patterns automatically.

    ---

    ### 6. Evaluation of Predictions
    Once a model makes predictions, we need to measure how good they are.  
    This means comparing the model’s predictions to the actual values.

    Let:
    - **actual** value = *y*
    - **predicted** value = ŷ (read as “y-hat”)

    The **error** is:

    > **error = actual − predicted = y − ŷ**

    Common ways to summarize how large the errors are:

    - **MAE (Mean Absolute Error):**  
      average of |y − ŷ| (the average size of the mistakes)
    - **MSE (Mean Squared Error):**  
      average of (y − ŷ)² (large mistakes count extra)
    - **RMSE (Root Mean Squared Error):**  
      the square root of MSE (in the same units as the data)
    - **MAPE (Mean Absolute Percentage Error):**  
      how large the errors are *relative* to the actual values, in %

    These measures help us compare models and choose the one that predicts best.

    For a bit more in-depth introduction, please also consider the following general papers:

    - [Machine learning: A primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC5905345/)
    - [Simple linear regression](https://www.nature.com/articles/nmeth.3627)
    - [Multiple linear regression](https://www.nature.com/articles/nmeth.3665)

# Motivation
Our tutorial aims to introduce aspects of statistical modelling and machine learning that are useful specifically for developing, evaluating and later operationalising forecasting models. Our pedagogical approach is to begin by introducing a very simple model in a simple setting, and then expanding both the model and the setting in a stepwise fashion. We emphasize interoperability and rigorous evaluation of models right from the start, as a way of guiding the development of more sophisticated models. In doing this, we follow a philosophy resembling what is known as agile development in computer science.
To facilitate interoperability and evaluation of models, we rely on the [Chap platform](https://dhis2-chap.github.io/chap-core), which enforces standards of interoperability already from the first, simple model. This interoperability allows models to be run on a broad data collection and be rigorously evaluated with rich visualisations of data and predictions already from the early phase.

# Making your first model and getting it into chap
Disease forecasting is a type of problem within what is known as spatiotemporal modelling in the field of statistics/ML. What this means is that the data have both a temporal and spatial reference (i.e. data of disease incidence is available at multiple subsequent time points, for different regions in a country), where observations that are close in time or space are usually considered more likely to be similar. In our case, we also have data both on disease and on various climate variables that may influence disease incidence.

Before going into the many challenges of spatiotemporal modelling, we recommend that you get the technical setup in place to allow efficient development and learning for the remainder of this tutorial. Although this can be a bit of a technical nuisance just now, it allows you to run your model on a variety of data inputs with rich evaluation now already, and it allows you to progress efficiently with very few technical wrinkles as the statistical and ML aspects become more advanced. To do this, please follow our minimalist example tutorial, which introduces an extremely oversimplified statistical model (linear regression of immediate climate effects on disease only), but shows you how to get this running in Chap. 
This minimalist tutorial is available both as Python and as R code:

- [Minimalist Example (Python)](https://github.com/dhis2-chap/minimalist_example) 
- [Minimalist Example (R)](https://github.com/dhis2-chap/minimalist_example_r) 


# Evaluating a model
The purpose of spatiotemporal modelling is to learn generalisable patterns that can be used to reason about unseen regions or about the future. Since our use case is an early warning system, our focus is on the latter, i.e. forecasting disease incidence ahead in time based on historic data for a given region. Therefore, we will focus on evaluating a model through its forecasting skill into the future.

A straightforward way to assess a forecasting model is to create and record forecasts for future disease development, wait to see how disease truly develops, and then afterwards compare the observed numbers to what was forecasted. This approach has two main limitations, though: 
it requires to wait through the forecast period to see what observations turn out to be
it only shows the prediction skill of the forecast model at a single snapshot in time - leaving a large uncertainty on how a system may be expected to behave if used to forecast new future periods. 

A popular and powerful alternative is thus to perform what is called backtesting or hindcasting: one pretends to be at a past point in time, providing a model exclusively with data that was available before this pretended point in time, making forecasts beyond that time point (for which no information was available to the model), and then assessing how close this forecast is to what happened after the pretended time point. When performed correctly, this resolves the both mentioned issues with assessing forecasts truly made into the future: 
Since observations after the pretended time point is already available in historic records, assessment can be performed instantaneously, and
one can choose several different pretended time points, reducing uncertainty of the estimated prediction skill  and also allowing to see variability in prediction skill across time. 

To be truly representative of true future use, it is crucial that the pretended time point for forecasting realistically reflects a situation where the future is not known. There are a myriad pitfalls in assessment setup that can lead to the assessment not respecting the principle of future information beyond the pretended time point not being accessible to models. This is discussed in more detail in the document ["Ensuring unbiased and operationally relevant assessments of climate-informed disease forecasting"](https://docs.google.com/document/d/1Hr7Wz4Yc4ZKZ6fsFJI_lpLO8d1SSxtfTeqByWZrQqFo/edit?tab=t.0).

Prediction skill can be measured in different ways. One simple way to measure this is to look at how far off the predictions are, on average, from the true values (known as mean absolute error, MAE). Other common measures are discussed later in this tutorial, after we have introduced further aspects of modelling. 
To make the most of the data we have, we often use a method called cross-validation. This means we repeatedly split the data into “past” and “future” parts at different time points. We then make forecasts for each split and check how accurate those forecasts are. This helps us see how well the model performs across different periods of time. To learn more, Wikipedia has a broad [introduction to this topic](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Cross_validation_for_time-series,_spatial_and_spatiotemporal_models), including specifics for time series models. 

Since we are here running our models through Chap, we can lean on an already implemented solution to avoid pitfalls and ensure a data-efficient and honest evaluation of the models we develop. Chap also includes several metrics, including MAE, and offers several convenient visualisations to provide insights on the prediction skill. 
To get a feeling for this, please follow the code-along tutorials on assessment with Chap. We recommend to start with our simple code-along [tutorial for how to split data and compute MAE](https://github.com/dhis2-chap/Assessment_example_singlepred ) on the pretended future. 

After getting a feeling for assessment, please continue with our code [tutorial on how to use the built-in chap evaluation functionality](https://github.com/dhis2-chap/Assessment_example_chap_compatible) to perform more sophisticated evaluation of any model (rather than implementing your own simple evaluation from scratch, as in the previous code-along tutorial).

With this evaluation setup in place, you should be ready to start looking at more sophisticated modelling approaches. For each new version of your model, evaluation helps you check that the new version is actually an improvement (and if so, in which sense, for instance short-term vs long-term, accuracy vs calibration, large vs small data set).

# Expanding your model to make it more meaningful
## Multiple regions
While it may in some settings be useful to forecast disease at a national level, it is often more operationally relevant to create forecasts for smaller regions within the country, for instance at district level. Therefore, a disease forecasting approach needs to relate to multiple districts in the code, to create forecasts per district. 
If a single model is trained across district data (ignoring the fact that there are different districts) and used to directly forecast disease without taking into account differences between districts in any way, it would forecast similar levels of disease across districts regardless of disease prevalence in each particular district. 
To see this more concretely please follow this [tutorial to see the errors](https://github.com/dhis2-chap/Assessment_example_multiregion) that the minimalist_example model (which ignores districts) makes for the two districts D1 and D2 with respectively high and low prevalence.

The simplest approach to creating meaningful region-level forecast is to simply train and forecast based on a separate model per region. Please follow the tutorial below (in Python or R version) to see an easy way of doing this in code: 

- [Minimalist Multiregion (Python)](https://github.com/dhis2-chap/minimalist_multiregion)
- [Minimalist Multiregion (R)](https://github.com/dhis2-chap/minimalist_multiregion_r)

However, such independent consideration of each region also has several weaknesses.
A main weakness is connected to the amount of data available to learn a good model. When learning a separate model per region, there may be a scarcity of data to fit a model of the desired complexity. This relates to a general principle in statistics concerning the amount of data available to fit a model versus the number of parameters in a model (see e.g. [this article](https://www.researchgate.net/publication/307526307_Points_of_Significance_Model_selection_and_overfitting)). More available data allows for a larger number of parameters and a more complex model. Compared to the case with separate models per region, just combining data across all regions into a single model where the parameters are completely independent between districts does not change the ratio of available data versus parameters to be estimated. However, if the parameters are dependent (for example due a spatial structure, i.e. similarities between regions), the effective number of parameters will be lower than the actual number. There is a trade-off between having completely independent parameters on one hand and, and forcing parameters to be equal across regions on the other hand. This is often conveyed as “borrowing strength” between regions. It is also related to the concept of the bias-variance tradeoff in statistics and machine learning (see e.g. [this Wikipedia article](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)), where dependency between parameter values across regions introduces a small bias in each region towards the mean across regions, while reducing variance of the parameter estimates due to more efficient use of data. 

As the disease context (including numbers of disease cases) can vary greatly between regions, the same type of model is not necessarily suited for all regions. However, taking this into account can be complex, especially if one wants to combine flexibility of models with the concept of borrowing strength mentioned above. It is thus often more practical to use a single model across regions, but ensure that this model is flexible enough to handle such heterogeneity across regions. 

## Lags and autoregressive effect
The incidence of disease today is usually affected by the incidence of disease in the recent past. This is for instance almost always the case with infectious disease, whether or not transmission is directly  human-to-human or via a vector like mosquitoes (e.g. Malaria and Dengue). Thus, it is usually advisable to include past cases of disease as a predictor in a model. This is typically referred to as including autoregressive effects in models.
 
Additionally, climate variables such as rainfall and temperature usually don’t have an instantaneous effect. Typically, there is no way that (for example) rainfall today would influence cases today or in the nearest days. Instead, heavy rainfall today could alter standing water, affecting mosquito development and behavior, with effects on reported Malaria cases being several weeks ahead. This means that models should typically make use of past climatic data to predict disease incidence ahead. The period between the time point that a given data value reflects and the time point of its effect is referred to as a lag. The effect of a predictor on disease may vary and even be opposite when considered at different lags. A model should usually include predictors at several different time lags, where the model aims to learn which time lags are important and what the effects are for a given predictor at each such lag.

Available data can be on different time resolutions. In some contexts, reported disease cases are available with specific time stamps, but more often what is collected is aggregated and made available at weekly or monthly resolution. The available data resolution influences how precisely variation in predictor effects across lag times can be represented. 

Basically, the influence of each predictor at each time lag will be represented by a separate parameter in a model. As discussed in the previous section on multiple regions, this can lead to too many parameters to be effectively estimated. It is thus common to employ some form of smoothing of the effects of a predictor across lags, with a reasoning similar to that of borrowing strength across regions (here instead borrowing strength across lag times).

At a practical level, model training is often assuming that all predictors to be used for a given prediction (including lagged variables) are available at the same row of the input data table. It is thus common to modify the data tables to make lagged predictors line up. 

To see a simple way of doing this in code, follow this tutorial in Python or R:

- [Minimalist Example Lag (Python)](https://github.com/dhis2-chap/minimalist_example_lag)
- [Minimalist Example Lag (R)](https://github.com/dhis2-chap/minimalist_example_lag_r)


# Expanding your model with uncertainty
Future disease development can never be predicted with absolute certainty - even for short time scales there will always be some uncertainty of how the disease develops. For a decision maker, receiving a point estimate (a single best guess) without any indication of uncertainty is usually not very useful. Consider a best guess of 100 cases in a region. The way to use this information would likely be very different if the model predicted a range from 95 to 105 vs a range from 0 to 10 000. Although the range provided by the model is itself uncertain, it still provides a useful indication of what can be expected, and it will be possible to evaluate its ability to report its own uncertainty (as will be explained in the next section). 

When doing statistical modelling, uncertainty arises from several distinct sources, many of which are particularly important in time series settings. First, observed data are often noisy or imperfect measurements of an underlying process. Measurement error, reporting delays, missing values, or later data revisions mean that the observed series does not exactly reflect the true system being modelled.

Even with perfect observations, the underlying process itself is usually stochastic, meaning that even if one had precise measurements of the current status and had learnt a perfectly correct model that involves these measured data, a variety of external factors would influence disease development in unpredictable ways. Additional uncertainty comes from estimating model parameters (relations between climate and disease) using often limited historical data. This uncertainty propagates into predictions. There is also uncertainty associated with model choice. Any model is a simplification of reality, and incorrect assumptions about lag structure, stationarity, seasonality, or linearity can lead to biased inference and misleading forecasts. This structural uncertainty is difficult to quantify but often dominates in practice.

Bayesian modelling provides a way to combine uncertainties from various sources in a coherent, principled way, making sure that the overall uncertainty is well represented. One aspect of this is that instead of estimating one particular value for each of its model parameters (e.g. for the relation between a given climate factor and the disease), it estimates a probability distribution for this value (referred to as the posterior distribution). Uncertainty intervals can then be based on the posterior distribution, for example using the interval between the 5% and 95% quantiles would give an interval with 90% probability of containing the true value of the parameter. Similarly, prediction intervals can be generated for the disease forecasts based on the uncertainty represented in the model itself (in its parameter distributions).

Many other modelling approaches (e.g. classical ARIMA and most machine learning models) are mainly focused on the trend of disease development - of predicting a point estimate (single value) for the expected (or most likely) number of disease cases ahead. Uncertainty is then often added on top of this prediction of expected value. A simple choice is to view the model predictions as representing the expected value of some probability distribution that captures the uncertainty of the forecast. For instance, one could view model forecasts as representing the expected value of a normal distribution with a fixed standard deviation that is estimated to capture the uncertainty as well as possible (under such an assumption of normality). Another more sophisticated alternative is to model this standard deviation itself, so that for every forecast, both the expected value and the uncertainty of the forecast is influenced by the available data.

Reporting forecasted point estimates is straightforward - it is just to report a single number per region and time period. Reporting a prediction that includes uncertainty is less trivial. If the uncertainty is captured by a given parametric distribution (like a plain normal distribution), one could simply report the parameters of this distribution (the mean and standard deviation in the case of the normal distribution). When uncertainty may follow different distributions, one needs a more flexible approach. One possibility is to report quantiles of this distribution - e.g. the 10% and 90% quantiles, which allows to see a 80% prediction interval. One could also report many of these. An even more generic approach to this is to report samples from the distribution, which allows the full distribution to be approximately reconstructed from the samples. It also allows any quantile to be approximated by simply sorting the samples and finding the value that a given proportion of samples are below. Due to its flexibility, this representation is often the preferred and chosen approach to represent predicted distributions, and is also what is used in the chap models when models report their forecasts back to the platform. 


## Evaluating predicted uncertainty

- The concept of uncertainty calibration (and why it is operationally important)

# Honest and sophisticated evaluation in more detail

- (see also separate manuscript about this..)
- Time series cross-validation.. (growing/rolling window)

# Relating to data quality issues, including missing data

# More realistic (Sophisticated) models

- BayesianHierarchicalModelling, including INLA, mentioning STAN
- Auto-regressive ML
- ARIMA and variants
- Deep learning

# Systematic evaluation on simulated data for debugging, understanding and stress-testing models

- Simulating from the model or something presumed close to the model - see if the model behaves as expected and how much data is needed to reach desired behavior..
- Simulate specific scenarios to stress-test models
- Develop more realistic simulations

# Selecting meaningful features (covariates) and data

# Further resources
Some general resources - not properly screened for now:

- [Introduction to Statistical Modeling, Hughes and Fisher, 2025](https://tjfisher19.github.io/introStatModeling/)
- [An Introduction to Statistical Learning, James et. al., 2021](https://www.statlearning.com/)
- [Intro to Time Series Forecasting](https://www.kaggle.com/code/iamleonie/intro-to-time-series-forecasting)
- [Time Series Forecasting: Building Intuition](https://www.kaggle.com/code/iamleonie/time-series-forecasting-building-intuition)
- [Regularization](https://www.nature.com/articles/nmeth.4014)
- [Importance of being uncertain](https://www.nature.com/articles/nmeth.2613)
