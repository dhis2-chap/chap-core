# Explain module

The chap_core/explain module is intended to provide functionality to estimate variable contribution for individual model predictions model-agnostically. Given an arbitrary model adhering to the CHAP external model standards, and a particular prediction to explain, it can return the weighting of input variables in the specified prediction.

Currently, the module leverages LIME to perform this analysis. LIME is a model-agnostic technique in explainability which trains an explainable surrogate model to behave similarly to the original model locally around the specified prediction. To achieve this, we produce a dataset of local input vectors by perturbing the original input vector randomly, and producing the target data by running these perturbed input vectors through the original model. We can then train the surrogate model using this new dataset of local inputs and predicted outputs, and may explain the original model by the behavior of the trained surrogate.

## User guide

The explain module is available through the command line interface (CLI). By calling "chap explain", followed by the correct arguments in the terminal, the module will produce and print the estimated variable contribution weighting. The arguments are:

- model_name: The name of the model to explain, either as a path or as a github url
- dataset_csv: Path to the csv file containing the training dataset. NOTE: Currently, the model is trained in the explain-pipeline, but this will be changed to loading a pre-trained model
- location: Name of the location on which to explain a prediction.
- horizon: Number of time steps (years, months or weeks) into the future for which to produce and explanation. This, together with location, determines the exact prediction to explain (e.g. the prediction of number of disease cases in Acre three weeks from now)
- historical_context_years: Number of time steps in the past for which to get a variable contribution weighting. Defaults to 6.
- surrogate_name: Short name of which surrogate model to use. Currently supports "ridge", "tree".
- threshold [Optional]: Number above which to count a simulated number of disease cases as an instance of a positive class. For instance, if set to 500, then a prediction of 400 disease cases will be counted as a negative class 0, while a prediction of 550 cases will be counted as a positive class 1. If not specified, the module will consider the value of the predicted cases rather than probability of exceeding a specified threshold.
- model_configuration_yaml [Optional]: Yaml file with the model configuration
- run_config [Optional]: Configuration of model run environment


Example of a run:

```bash
chap explain --model_name https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6 --dataset_csv example_data/nicaragua_weekly_data.csv --location boaco --horizon 3
```
Example output:

```bash
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: Coefficients:
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: rainfall_fut_3: -0.5078
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: rainfall_fut_2: -0.2607
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: population_fut_3: -0.2298
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: mean_temperature_fut_3: -0.2086
2026-01-28 17:27:58,668 INFO chap_core.explainability.lime: mean_temperature_fut_1: +0.1438
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t-1: -0.1233
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: rainfall_fut_1: +0.1167
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t-5: +0.0487
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_fut_1: -0.0469
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t-3: -0.0358
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t: -0.0343
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t-2: -0.0290
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_fut_2: -0.0187
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_t-4: -0.0153
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: mean_temperature_fut_2: -0.0130
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_t: +0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: rainfall_t-4: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: rainfall_t-3: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime:   rainfall_t: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_t-3: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: rainfall_t-5: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_t-1: -0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_t-5: +0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: rainfall_t-2: +0.0000
2026-01-28 17:27:58,669 INFO chap_core.explainability.lime: population_t-4: -0.0000
2026-01-28 17:27:58,670 INFO chap_core.explainability.lime: population_t-2: +0.0000
2026-01-28 17:27:58,670 INFO chap_core.explainability.lime: rainfall_t-1: +0.0000
```

## Code guide

The explainability module currently consists of two files; lime.py and surrogate.py.

### lime.py

lime.py contains the implementation of the full pipeline for performing the LIME algorithm. The central function is explain(...), which executes this pipeline and produces the results. Alongside this function are many supporting functions, separated for code clarity and ease of maintenance.

#### **explain(...)**

explain(...) executes the full pipeline from producing the perturbed local dataset, transforming from interpretable vectors into model input, to training the surrogate model and printing the resulting variable contribution weighting. The function currently handles the nature of time series predictions by considering only a particular step in the time series as the target; treating all predictions before that as input on equal footing with the actual dataset, and ignoring any predictions after it. 

Currently the surrogate model is a class handled in surrogate.py for easy maintenance. Eventually, the vector perturbation will also be handled by a plug-and-play set of class instances.

    Args:
    - model (ExternalModel): A trained predictor on which to generate explanation
    - dataset (DataSet): The dataset on which to perturb
    - location (str): The location on which to explain
    - horizon (int): The number of time steps into the future on which to explain
    - granularity (int): Number of time steps in advance to find weighting on (default: 4)
    - surrogate_name (str): The model used as explainable surrogate - one of ["ridge", "tree"] (Default ridge)
    - threshold (Optional float): The threshold above which to count as positive observation

    The function doesn't return anything, but prints the variable contribution weighting.


#### **prob_thresh(...)**

prob_thresh(...) takes a dataset of simulated runs from the model, where each simulation run produces a predicted value of disease cases per time step, and returns a dictionary of the probability of exceeding a certain threshold of number of disease cases per time step. If nearly all simulated runs predict a number of disease cases above the threshold T at time step D, then prob_thresh will return a value for D close to 1. On the other hand, if only half of the simulated runs predicts a value above threshold, prob_thresh will return 0.5 at D.

    Args:
        - data (DataSet): A dataset of range of sampled predictions
        - threshold (float): The threshold of which to calculate the probability of exceeding

    Returns:
        Dictionary of probability over time


#### **avg_samples(...)**

If no threshold is supplied when calling explain(...), the pipeline will use avg_samples(...) to transform a range of simulated runs from the model into a single value by averaging across all runs.
    
    Args:
        data (DataSet): A dataset of range of sampled predictions

    Returns:
        Dictionary of average over time



#### **is_constant(...)**

Support function to check whether the values of a column in a dataset remain constant across time steps. If it does, there is no reason to include the lagged values as input variables.
    
    Args:
        window (pandas.DataFrame): Dataframe with data within boundries of granularity
        features (list(str)): List of feature names of dataset
        num_steps (int): Number of time steps ito check constantness over
    
    Return:
        bool of whether value is constant


#### **build_original_vector(...)**

build_original_vector(...) produces an interpretable vector of the original prediction input.
    
    Args:
        window (pandas.DataFrame): Dataframe with data within boundries of granularity
        window (pandas.DataFrame): Dataframe with future data within boundries of horizon
        features_hist (list(str)): List of feature names of historical dataset
        features_fut (list(str)): List of feature names of future dataset
        horizon (int): Number of time steps in future to include in vector
        granularity (int): Number of time steps in advance to include in vector
    
    Returns:
        Dictionary of feature names and value


