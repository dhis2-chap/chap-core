## Code documentation

TODO: Very outdated





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


