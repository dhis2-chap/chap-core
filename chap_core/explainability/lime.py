
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from chap_core.climate_predictor import get_climate_predictor
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.explainability.surrogate import RidgeSurrogate, SurrogateModel, TreeSurrogate
from chap_core.model_spec import _non_feature_names
from chap_core.models.external_model import ExternalModel
from chap_core.models.utils import get_model_from_directory_or_github_url
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import PeriodRange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def prob_thresh(
    data: DataSet, 
    threshold: float
) -> Dict:
    """
    Convert range of simulated runs into probability of seeing a higher result than threshold
    
    Args:
        data (DataSet): A dataset of range of sampled predictions
        threshold (float): The threshold of which to calculate the probability of exceeding

    Returns:
        Dictionary of probability over time
    """
    dataframe = data.to_pandas()
    # Only keep the sample columns
    sample_cols = dataframe.filter(regex=r"^sample_\d+$").columns
    thresh_prob_mask = dataframe[sample_cols] > threshold
    # Probability of exceeding threshold estimated as mean of exceeding mask
    thresh_prob = thresh_prob_mask.mean(axis=1)
    tp_len = len(dataframe["time_period"]); prb_len = len(thresh_prob)
    assert tp_len == prb_len, f"Error: Length of time period {tp_len} is not equal to length of probabilities {prb_len}"
    return dict(zip(dataframe["time_period"], thresh_prob))

def avg_samples(
    data: DataSet, 
) -> Dict:
    """
    Average range of simulated runs
    
    Args:
        data (DataSet): A dataset of range of sampled predictions

    Returns:
        Dictionary of average over time
    """
    dataframe = data.to_pandas()
    # Only keep the sample columns
    sample_cols = dataframe.filter(regex=r"^sample_\d+$").columns
    mean_vals = dataframe[sample_cols].mean(axis=1)
    tp_len = len(dataframe["time_period"]); val_len = len(mean_vals)
    assert tp_len == val_len, f"Error: Length of time period {tp_len} is not equal to length of values {val_len}"
    return dict(zip(dataframe["time_period"], mean_vals))

def is_constant(
    window: pd.DataFrame,
    feature_name: str,
    num_steps: int,
    count_down: bool = True
) -> bool:
    """
    Check whether feature varies with time

    Args:
        window (pandas.DataFrame): Dataframe with data within boundries of granularity
        features (list(str)): List of feature names of dataset
        num_steps (int): Number of time steps ito check constantness over
    
    Return:
        bool of whether value is constant
    """
    value_set = set()
    for i in range(num_steps):
        pos = [-(i+1) if count_down else (i)][0]
        value_set.add(float(window[feature_name].iloc[pos]))
    if (len(value_set) <= 1): 
        return True
    return False



def build_original_vector(
    window: pd.DataFrame,
    future: pd.DataFrame,
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    granularity: int
) -> Dict[str, float | str]:
    """
    Create original information vector for producing perturbed input

    Args:
        window (pandas.DataFrame): Dataframe with data within boundries of granularity
        window (pandas.DataFrame): Dataframe with future data within boundries of horizon
        features_hist (list(str)): List of feature names of historical dataset
        features_fut (list(str)): List of feature names of future dataset
        horizon (int): Number of time steps in future to include in vector
        granularity (int): Number of time steps in advance to include in vector
    
    Returns:
        Dictionary of feature names and value
    """
    x0 = {}
    for name in features_hist:
        if is_constant(window, name, granularity):  # If features doesn't vary with time, add once
            x0[name] = float(window[name].iloc[-1])
        else:
            for i in range(granularity):  # Add an entry for each time step in granularity
                name_with_time_step = name + "_t" + [f"-{i}" if i>0 else ""][0]
                x0[name_with_time_step] = float(window[name].iloc[-(i+1)])
    for name in features_fut:
        if is_constant(future, name, horizon, count_down=False):
            x0[name] = float(future[name].iloc[0])
        else:
            for i in range(horizon):
                name_with_time_step = name + "_fut_" + str(i+1)
                x0[name_with_time_step] = float(future[name].iloc[i])

    return x0


def linear_shift(
    rng: random.Random,
    feature_name: str,
    lagged_dict: Dict[int, float]
):
    can_be_negative = (feature_name in ["temperature"])  # TODO: Shouldn't be hardcoded
    new_vals = []
    vals = [v for v in lagged_dict.values()]
    avg_val = sum(vals)/len(vals)
    shift = rng.uniform(-avg_val, avg_val)
    for lag in sorted(lagged_dict.keys(), reverse=True):
        new_val = lagged_dict[lag] + shift
        if not can_be_negative:
            new_val = max(0, new_val)
        new_vals.append(new_val)
    return new_vals

def data_sample(
    rng: random.Random,
    hist_df: pd.DataFrame,
    feature_name: str,
    lagged_dict: Dict[int, float]
):
    """
    Perturbs lagged features by sampling randomly from existing dataset

    Args:
        rng (random Random): RNG object to preserve seed
        hist_df (pandas DataFrame): Dataframe of original historical data
        feature_name (str): The lagged feature of which to sample
        lagged_dict (dict(int, float)): Original time keys and values of lagged feature

    Return:
        List of perturbed vectors
    """

    time_range = len(lagged_dict.keys())
    locations = hist_df["location"].dropna().unique().tolist()

    for _ in range(3): # Retries twice if selected window is too narrow
        loc = rng.choice(locations)
        loc_df = hist_df[hist_df["location"] == loc].copy()  # TODO: Currently only one location in df
        if len(loc_df) < time_range:
            continue
        
        max_start = len(loc_df) - time_range
        if max_start < 0:
            continue
        start = rng.randrange(0, max_start + 1)
        window = loc_df.iloc[start : start + time_range]

        vals = window[feature_name].astype(float).tolist()
        if any(pd.isna(v) for v in vals):
            continue

        return vals[::-1]
    
    # Fallback
    lag_ints = sorted(lagged_dict.keys(), reverse=True)
    return [float(lagged_dict[lag]) for lag in lag_ints]



def perturb_vector(
    dataset: pd.DataFrame,
    orig_vector: Dict,
    num_perturbations: int,
    mutation_rate: float,
    seed: int = None,
    lagged_feature_strategy: str = "dataset_sample"  # TODO: Make enum? Also investigate effectiveness
) -> List[Dict]:
    """
    Perturb original vector to get local variations

    Args:
        dataset (pandas DataFrame): Historical data of model
        orig_vector (dict): Dictionary of original features and values
        num_perturbations (int): Number of perturbed vectors to create
        mutation_rate (float): Probability of changing a particular value
        seed (int): Seed for rng
        lagged_feature_strategy (str): How to perturb lagged features. Can be one of
                    [random, linear_shift, dataset_sample] (Default: dataset_sample)

    Return:
        List of perturbed vectors
    """
    rng = random.Random(seed)
    perturbations = []
    lagged_pattern = re.compile(r"^(?P<name>.+?)_t(?P<lag>-\d+)?$")
    for _ in range(num_perturbations):
        vec = {}
        feature_groupings = {}
        for key, value in orig_vector.items():
                m = lagged_pattern.match(key)
                if not m:
                    # Not lagged feature
                    if (rng.random() < mutation_rate 
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)):  # TODO: Handle categorical?
                        new_val = rng.gauss(float(value), 1.0)  # TODO: Functionality to vary sampling?
                        vec[key] = new_val
                    else:
                        vec[key] = value
                    continue
                name = m.group("name")
                lag_str = m.group("lag")
                lag = 0 if lag_str is None else int(lag_str)
                if name not in feature_groupings.keys():
                    feature_groupings[name] = {}
                feature_groupings[name][lag] = value

        for name, series in feature_groupings.items():  # TODO: Is series always sorted
            lag_ints = sorted(series.keys(), reverse=True)
            if rng.random() < mutation_rate and any(isinstance(value, (int, float)) for value in series.values()):
                match lagged_feature_strategy:
                    case "random":
                        raise NotImplementedError() #lagged_random(rng, name, series)
                    case "linear_shift":
                        new_vals = linear_shift(rng, name, series)
                    case "dataset_sample":
                        new_vals = data_sample(rng, dataset, name, series)
                    case _:
                        raise ValueError(f"Illegal lagged feature strategy: {lagged_feature_strategy}")
                # TODO: Ensure order of new_vals is not altered compared to series
                vec[f"{name}_t"] = new_vals[0]
                for i in range(1, len(lag_ints)):
                    vec[f"{name}_t{lag_ints[i]}"] = new_vals[i]
            else:
                vec[f"{name}_t"] = series[lag_ints[0]]
                for i in range(1, len(lag_ints)):
                    vec[f"{name}_t{lag_ints[i]}"] = series[lag_ints[i]]
        perturbations.append(vec)

    return perturbations



def convert_vector_to_dataset(
    perturbation: Dict, 
    hist_df: pd.DataFrame, 
    fut_df: pd.DataFrame,
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    granularity: int,
    hist_type: Type,
    fut_type: Type,
) -> Tuple[DataSet, DataSet]:
    """
    Convert interpretable vector back into full dataset
    
    Args:
        perturbation (dict): The perturbed interpretable vector
        hist_df (pandas DataFrame): The original historic dataset
        fut_df (pandas DataFrame): The future weather datset
        features_hist (list(str)): List of feature names of historical dataset
        features_fut (list(str)): List of feature names of future dataset
        horizon (int): Number of future time steps the vector encompasses
        granularity (int): Number of historic time steps the vector encompasses
        location (str): Geographical location of data,
        hist_type (type): Dataclass of historical data
        fut_type (type): Dataclass of future data
        
    Return:
        Historic dataset with perturbed values inserted
        Future dataset with perturbed values inserted
    """
    # TODO: Ensure dataset only contains "location"?
    # Historic data insertions
    hist_idx0 = len(hist_df) - granularity
    for feat in features_hist:
        has_lagged = (f"{feat}_t" in perturbation) or any(
            f"{feat}_t-{lag}" in perturbation for lag in range(1, granularity)
        )
    
        if has_lagged:
            for lag in range(granularity):
                key = f"{feat}_t" if lag==0 else f"{feat}_t-{lag}"
                if key in perturbation and feat in hist_df.columns:
                    row = (len(hist_df) - 1) - lag
                    hist_df.loc[row, feat] = float(perturbation[key])
        else:
            if feat in perturbation and feat in hist_df.columns:
                hist_df.loc[hist_idx0:, feat] = float(perturbation[feat])
    
    # Future data insertions
    for feat in features_fut:
        has_steps = any(f"{feat}_fut_{i+1}" in perturbation for i in range(horizon))

        if has_steps:
            for i in range(horizon):
                key = f"{feat}_fut_{i+1}"
                if key in perturbation and feat in fut_df.columns:
                    fut_df.loc[i, feat] = float(perturbation[key])
        else:
            if feat in perturbation and feat in fut_df.columns:
                fut_df.loc[: horizon - 1, feat] = float(perturbation[feat])

    new_hist = DataSet.from_pandas(hist_df, dataclass=hist_type)
    new_fut = DataSet.from_pandas(fut_df, dataclass=fut_type)

    return new_hist, new_fut



def build_X_y(
    results: List[Tuple[Dict, float | int]], 
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build input matrix and target vector for linear model training
    
    Args:
        results (List[Tuple[Dict, float | int]]): List of perturbed vectors and corresponding targets
        feature_names (List[str]): List of all feature names in perturbed vectors
    
    Returns:
        Tuple of numpy arrays
    """
    X, y = [], []
    for vec, prob in results:
        X.append([float(vec[f]) for f in feature_names])
        y.append(float(prob))
    return np.asarray(X), np.asarray(y)

def produce_lime_dataset(
    model: ExternalModel, 
    dataset: pd.DataFrame,
    future_weather: pd.DataFrame,
    perturbations: List[Dict],
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    granularity: int,
    location: str,
    threshold: Optional[float] = None,
    hist_type: Type = None,
    fut_type: Type = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Produce training input and target dataset for LIME linear surrogate model
    
    Args:
        model (ExternalModel): Model used for producing target values
        dataset (pandas DataFrame): Dataset of historic data to perturb
        future_weather (pandas DataFrame): Dataset of future weather to perturb
        perturbations (List[Dict]): List of perturbed input vectors
        features_hist (List[str]): The feature names in the historic dataset
        features_fut (List[str]): The feature names of the future dataset
        horizon (int): The number of time steps in the past for which we want detailed explanations
        granularity (int): The number of time steps into the future for which we want an explanation
        location (str): Geographical location of the data
        threshold (Optional float): The threshold of disease cases above which counted as positive class (Default None)
        hist_type (type): Dataclass of historical data (Default None)
        fut_type (type): Dataclass of future data (Default None)

    Returns:
        Tuple of numpy arrays and feature name list
    """
    results = []
    # Batch prediction is quicker than individual predictions
    # Key different perturbations by pseudo-location, then batch predict
    full_hist_dict = {}
    full_fut_dict = {}
    pert_map = {}
    for i, pb in enumerate(perturbations):
        loc_id = f"pb_{i}"
        pert_map[loc_id] = pb
        # TODO: Should refactor external_model a bit, avoid reduplication on adapt especially
        new_hist, new_fut = convert_vector_to_dataset(
            pb, 
            dataset, 
            future_weather, 
            features_hist, 
            features_fut, 
            horizon, 
            granularity, 
            hist_type,
            fut_type
        )
        full_hist_dict[loc_id] = new_hist.get_location(location)
        full_fut_dict[loc_id] = new_fut.get_location(location)

    hist_combined = DataSet(full_hist_dict, polygons=None) 
    fut_combined = DataSet(full_fut_dict, polygons=None)
    
    pred_v = model.predict(hist_combined, fut_combined)

    for ds in pred_v.iter_locations():
        loc_name = next(iter(ds.locations()))
        if loc_name in pert_map:
            pb_vec = pert_map[loc_name]
            if threshold is not None:  # TODO: Should probably be a class eventually
                vals = prob_thresh(ds, threshold=threshold)
            else:
                vals = avg_samples(ds)
            # Extract most recent "prob" as horizon value
            latest = max(vals.keys())
            latest_prob = vals[latest]
            results.append((pb_vec, latest_prob))

    # TODO: This is brittle, especially as regards pairing it up with coefficients later
    feature_names = [k for k in perturbations[0].keys() if k not in _non_feature_names]
    X, y = build_X_y(results, feature_names)
    return X, y, feature_names


def logit(p):
    return np.log(p / (1.0 - p))

def lime_weights_from_X(
    X: np.ndarray,
    x0_row: np.ndarray,
    kernel_width: float = 1.0,
) -> np.ndarray:
    """
    RBF kernel weights based on distance to x0 in standardized space.

    Args:
        X (np.ndarray): Perturbed dataset
        x0_row (np.ndarray): Original input vector
        kernel_width (float): Width of the RBF kernel (default 1.0)
    """
    # Standardize for distance
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    x0s = scaler.transform(x0_row.reshape(1, -1))[0]

    d = np.linalg.norm(Xs - x0s, axis=1)
    w = np.exp(-(d**2) / (kernel_width**2))
    return w



def instantiate_model(name: str):
    match name.lower():
        case "ridge":
            return RidgeSurrogate()
        case "tree":
            return TreeSurrogate()
        case _:
            raise ValueError(f"Unknown surrogate model: {name}")



def explain(
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    granularity: int = 4,
    num_perturbations: int = 40,  # TODO: Include in endpoint
    surrogate_name: str = "ridge",
    threshold: Optional[float] = None
):
    """
    Model-agnostic function to supply variable contribution weighting for specific prediction
    
    Args:
        model (ExternalModel): A trained predictor on which to generate explanation
        dataset (DataSet): The dataset on which to perturb
        location (str): The location on which to explain
        horizon (int): The number of time steps into the future on which to explain
        granularity (int): Number of time steps in advance to find weighting on (default: 4)
        num_perturbations (int): Number of generated perturbed variations of input vector (default 200)
        surrogate_name (str): The model used as explainable surrogate - one of ["ridge", "tree"] (default ridge)
        threshold (Optional float): The threshold above which to count as positive observation (default None)
    """
    # Initial input safety checks
    assert horizon > 0, f"Horizon must be positive; received horizon={horizon}"
    assert location in dataset.locations(), f"Location {location} not found in dataset"
    # TODO: Assert granularity < len(dataset)

    delta = dataset.period_range[0].time_delta
    prediction_range = PeriodRange(
        dataset.end_timestamp,
        dataset.end_timestamp + delta * horizon,
        delta,
    )

    climate_predictor = get_climate_predictor(dataset)
    future_weather = climate_predictor.predict(prediction_range)
    
    # Isolate dataset to selected location
    dataset_loc = dataset.filter_locations([location])  # TODO: Just use dataset[location]?
    future_weather = future_weather.filter_locations([location])
    hist_type = dataset_loc[location].__class__
    fut_type = future_weather[location].__class__

    # Sort by dates, and extract feature names
    hist_df = dataset_loc.to_pandas().sort_values("time_period").reset_index(drop=True)
    future_df = future_weather.to_pandas().sort_values("time_period").reset_index(drop=True)
    assert len(future_df) >= horizon, f"Need at least {horizon} future steps, got {len(future_df)}"
    # Isolate data from last (granularity) time steps
    window = hist_df.iloc[-granularity:].reset_index(drop=True)

    features_hist = [
        fn 
        for fn in dataset_loc.field_names() 
        if fn not in _non_feature_names
    ]

    features_fut = [
        fn
        for fn in future_weather.field_names() 
        if fn not in _non_feature_names
    ]

    # Build original vector around which to generate perturbed vectors
    x0 = build_original_vector(
        window, 
        future_df, 
        features_hist, 
        features_fut, 
        horizon, 
        granularity
    )

    # Create perturbed variations
    # TODO: When, where and how to decide number of perturbations and mr?
    perturbations = perturb_vector(hist_df, x0, num_perturbations=num_perturbations, mutation_rate=0.3)
    
    # Threshold picked arbitrarily in example
    X, y, feature_names = produce_lime_dataset(
        model, 
        hist_df,
        future_df,
        perturbations,
        features_hist,
        features_fut,
        horizon,
        granularity,
        location,
        threshold,
        hist_type,
        fut_type
    )

    x0_row = np.array([float(x0[f]) for f in feature_names], dtype=float)

    weights = lime_weights_from_X(X, x0_row, kernel_width=1.0)

    if threshold is not None:  # Transform z based on target type
        eps = 1e-4
        y_clip = np.clip(y, eps, 1-eps)
        z = logit(y_clip)
    else:
        z = np.log1p(y) # Log transform

    surrogate_model = instantiate_model(surrogate_name)
    surrogate_model.fit(X, z, weights)
    results = surrogate_model.explain(feature_names)

    logger.info("Coefficients:")
    for name, c in results.as_sorted():
        logger.info(f"{name:>12}: {c:+.4f}")




if __name__ == "__main__":
    model_name = 'https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6'#'https://github.com/chap-models/chap_auto_ewars'
    estimator = get_model_from_directory_or_github_url(model_name)
    dataset = ISIMIP_dengue_harmonized['vietnam']
    predictor = estimator.train(dataset)


    explain(
        predictor,
        dataset,
        location="BinhDinh",
        horizon=3,
        granularity=5,
    )