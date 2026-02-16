
import logging
import random
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from chap_core.climate_predictor import get_climate_predictor
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.explainability.surrogate import RidgeSurrogate, TreeSurrogate
from chap_core.explainability.segment import SegmentationModel, UniformSegmentation
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
    num_steps: int | None = None,
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
    series = window[feature_name]

    if num_steps is not None:
        series = series.iloc[-num_steps:] if count_down else series.iloc[:num_steps]

    return series.nunique(dropna=False) <= 1



def build_original_vector(
    segmenter: SegmentationModel,
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    granularity: int
) -> Dict[str, float | str | Dict[str, float]]:
    """
    Create original information vector for producing perturbed input

    Args:
        hist_df (pandas.DataFrame): Complete history dataframe
        fut_df (pandas.DataFrame): Dataframe with future data
        features_hist (list(str)): List of feature names of historical dataset
        features_fut (list(str)): List of feature names of future dataset
        horizon (int): Number of time steps in future to include in vector
        granularity (int): Number of time steps in advance to include in vector
    
    Returns:
        Dictionary of feature names and value
    """
    x0 = {}
    for name in features_hist:
        if is_constant(hist_df, name):  # If features doesn't vary with time, add once
            x0[name] = float(hist_df[name].iloc[-1])
        else:
            lagged_dict = segmenter.segment(hist_df[name])
            x0[name] = lagged_dict
    for name in features_fut:  # Future features are not segmented (TODO?)
        if is_constant(fut_df, name, horizon, count_down=False):
            x0[name] = float(fut_df[name].iloc[0])
        else:
            for i in range(horizon):
                name_with_time_step = name + "_fut_" + str(i+1)
                x0[name_with_time_step] = float(fut_df[name].iloc[i])

    return x0


def linear_shift(
    rng: random.Random,
    feature_name: str,
    segment: List[float]
) -> List[float]:
    """
    Perturbs lagged features by shifting them linearly (and clipping if illegaly negative)

    Args:
        rng (random Random): RNG object to preserve seed
        feature_name (str): The lagged feature of which to sample
        segment (list(float)): Original segment values

    Return:
        List of perturbed values
    """
    can_be_negative = (feature_name in ["temperature"])  # TODO: Shouldn't be hardcoded    
    if not segment: return []
    
    avg_val = sum(segment) / len(segment)
    shift = rng.uniform(-avg_val, avg_val)

    new_segment = []
    for val in segment:
        new_val = val + shift
        if not can_be_negative:
            new_val = max(0.0, new_val)
        new_segment.append(new_val)
    return new_segment

def data_sample(
    rng: random.Random,
    hist_df: pd.DataFrame,
    feature_name: str,
    length: int
) -> List[float]:
    """
    Perturbs lagged features by sampling randomly from existing dataset

    Args:
        rng (random Random): RNG object to preserve seed
        hist_df (pandas DataFrame): Dataframe of original historical data
        feature_name (str): The lagged feature of which to sample
        length: Length of data to sample

    Return:
        List of sampled values
    """

    locations = hist_df["location"].dropna().unique().tolist()

    for _ in range(3): # Retries twice if selected window is too narrow
        loc = rng.choice(locations)
        loc_df = hist_df[hist_df["location"] == loc]

        if len(loc_df) < length:
            continue

        max_start = len(loc_df) - length            
        start = rng.randrange(0, max_start + 1)
        vals = loc_df.iloc[start : start + length][feature_name].astype(float).tolist()

        if any(pd.isna(v) for v in vals):
            continue

        return vals
    
    # Fallback
    return [0.0] * length



def perturb_vector(
    dataset: pd.DataFrame,
    orig_vector: Dict,
    num_perturbations: int,
    mutation_rate: float,
    seed: int = None,
    lagged_feature_strategy: str = "dataset_sample"  # TODO: Make enum? Also investigate effectiveness
) -> Tuple[List[Dict], List[Dict]]:
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
    perturbations: List[Dict] = []
    perturbation_masks: List[Dict] = []
    for _ in range(num_perturbations):
        vec = {}
        pert_mask = {}
        for key, val in orig_vector.items():
            # Handle static features
            if isinstance(val, (float, int)):
                if rng.random() < mutation_rate:
                    vec[key] = rng.gauss(val, 1.0)
                    pert_mask[key] = 0
                else:
                    vec[key] = val
                    pert_mask[key] = 1
                continue

            if isinstance(val, dict):
                new_dict = {}
                lag_mask = {}
                for lag, segment in val.items():
                    if rng.random() < mutation_rate:
                        lag_mask[lag] = 0
                        match lagged_feature_strategy:
                            case "linear_shift":
                                new_dict[lag] = linear_shift(rng, key, segment)
                            case "dataset_sample":
                                new_dict[lag] = data_sample(rng, dataset, key, len(segment))
                            case _:
                                raise ValueError(f"Unknown strategy: {lagged_feature_strategy}")
                    else:
                        lag_mask[lag] = 1
                        new_dict[lag] = segment
                vec[key] = new_dict
                pert_mask[key] = lag_mask

        perturbations.append(vec)
        perturbation_masks.append(pert_mask)
    return perturbations, perturbation_masks



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

    hist_df = hist_df.copy()
    fut_df = fut_df.copy()

    # These loops stop futurewarning for casting columns
    for feat in features_hist:
        if feat in hist_df.columns:
            hist_df[feat] = hist_df[feat].astype(float)
            
    for feat in features_fut:
        if feat in fut_df.columns:
            fut_df[feat] = fut_df[feat].astype(float)
    
    for feat in features_hist:
        if feat not in perturbation:
            continue

        val = perturbation[feat]
        if isinstance(val, dict):
            num_segments = len(val)
            total_len = len(hist_df)
            segment_len = total_len // num_segments

            for lag_key, segment_vals in val.items():
                lag = lag_key
                real_idx = (num_segments - 1) - lag
                start_idx = real_idx * segment_len
                end_idx = start_idx + len(segment_vals)

                col_idx = hist_df.columns.get_loc(feat)
                hist_df.iloc[start_idx:end_idx, col_idx] = segment_vals
        else:
            hist_df[feat] = float(val)
    
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


def flatten_vector(
    vector: Dict
) -> Dict[str, int]:
    """
    Flattens a perturbation vector mask into a flat dict.
    
    Args:
        vector (dict): The vector to be flattened

    Returns:
        The flattened vector as dictionary of string to int
    """
    flat = {}
    for key, val in vector.items():
        if isinstance(val, dict):
            for lag, segment_data in val.items():
                    flat[f"{key}_lag_{lag}"] = segment_data
        elif isinstance(val, (int, float)):
            flat[key] = val
    return flat

def produce_lime_dataset(
    model: ExternalModel, 
    dataset: pd.DataFrame,
    future_weather: pd.DataFrame,
    perturbations: List[Dict],
    perturbation_masks: List[Dict],
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    granularity: int,
    location: str,
    threshold: Optional[float] = None,
    hist_type: Type = None,
    fut_type: Type = None,
    chunk_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Produce training input and target dataset for LIME linear surrogate model
    
    Args:
        model (ExternalModel): Model used for producing target values
        dataset (pandas DataFrame): Dataset of historic data to perturb
        future_weather (pandas DataFrame): Dataset of future weather to perturb
        perturbations (List[Dict]): List of perturbed input vectors
        perturbation_masks (List[Dict]): List of perturbed features mapped to binary
        features_hist (List[str]): The feature names in the historic dataset
        features_fut (List[str]): The feature names of the future dataset
        horizon (int): The number of time steps in the past for which we want detailed explanations
        granularity (int): The number of time steps into the future for which we want an explanation
        location (str): Geographical location of the data
        threshold (Optional float): The threshold of disease cases above which counted as positive class (Default None)
        hist_type (type): Dataclass of historical data (Default None)
        fut_type (type): Dataclass of future data (Default None)
        chunk_size (int): Size of prediction chunks

    Returns:
        Tuple of numpy arrays and feature name list
    """
    results: List[Tuple[Dict[str, float], str]] = []
    # Batch prediction is quicker than individual predictions
    # Key different perturbations by pseudo-location, then batch predict
    for i in range(0, len(perturbations), chunk_size):
        chunk = perturbations[i : i + chunk_size]
        full_hist_dict = {}
        full_fut_dict = {}
        pert_map = {}
        logger.info(f"Processing prediction chunk {i//chunk_size + 1} ({len(chunk)} perturbations)...")
        chunk_masks = perturbation_masks[i : i + len(chunk)]
        for j, pb in enumerate(chunk):
            loc_id = f"pb_{j}"
            pert_map[loc_id] = chunk_masks[j]
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
                flat_vec = flatten_vector(pb_vec)
                results.append((flat_vec, latest_prob))
    
    if not results:
        raise ValueError("No results generated")

    first_vec = results[0][0]
    feature_names = [k for k in first_vec.keys() if k not in _non_feature_names]
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
    granularity: int = 5,
    num_perturbations: int = 300,  # TODO: Include in endpoint
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

    segmenter = UniformSegmentation(num_segments=granularity) # TODO: Route to selected, granularity is num_segments

    # Build original vector around which to generate perturbed vectors
    x0 = build_original_vector(
        segmenter,
        hist_df, 
        future_df, 
        features_hist, 
        features_fut, 
        horizon, 
        granularity
    )

    full_dataset_df = dataset.to_pandas()

    # Create perturbed variations
    # TODO: When, where and how to decide number of perturbations and mr?
    perturbations, perturbation_masks = perturb_vector(full_dataset_df, x0, num_perturbations=num_perturbations, mutation_rate=0.2)
    
    # Threshold picked arbitrarily in example
    X, y, feature_names = produce_lime_dataset(
        model, 
        hist_df,
        future_df,
        perturbations,
        perturbation_masks,
        features_hist,
        features_fut,
        horizon,
        granularity,
        location,
        threshold,
        hist_type,
        fut_type
    )
    
    len_vector = len(X[0])
    x0_row = np.ones(X.shape[1], dtype=float)

    kw = 0.75 * np.sqrt(X.shape[1])  # This can be automatically inferred per paper "initial step towards stable" et.c.; magic number for now
    weights = lime_weights_from_X(X, x0_row, kernel_width=kw)

    if threshold is not None:  # Transform z based on target type
        eps = 1e-4
        y_clip = np.clip(y, eps, 1-eps)
        z = logit(y_clip)
    else:
        z = np.log1p(y) # Log transform

    surrogate_model = instantiate_model(surrogate_name)
    surrogate_model.fit(X, z, weights)
    results = surrogate_model.explain(feature_names)

    z_hat = surrogate_model.predict(X)
    r2 = r2_score(z, z_hat, sample_weight=weights)
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    logger.info(f"Surrogate weighted R2={r2:.3f}, effective N={n_eff:.1f}, p={X.shape[1]}")

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