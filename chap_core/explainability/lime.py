import logging
from multiprocessing import Value
import random
from typing import Dict, List, Tuple, Type, Any

import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from chap_core.climate_predictor import get_climate_predictor
from chap_core.explainability.surrogate import *
from chap_core.explainability.segment import *
from chap_core.explainability.perturb import *
from chap_core.explainability.distance import *
from chap_core.explainability.plot import plot_importance
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
    # TODO: Reduce number of samples in prediction?
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
) -> Tuple[Dict, Dict[str, Indices]]:
    """
    Create original information vector for producing perturbed input

    Args:
        hist_df (pandas.DataFrame): Complete history dataframe
        fut_df (pandas.DataFrame): Dataframe with future data
        features_hist (list(str)): List of feature names of historical dataset
        features_fut (list(str)): List of feature names of future dataset
        horizon (int): Number of time steps in future to include in vector
    
    Returns:
        Dictionary of feature names and value
    """
    x0 = {}
    feat_indices = {}
    for name in features_hist:
        if is_constant(hist_df, name):  # If features doesn't vary with time, add once
            x0[name] = float(hist_df[name].iloc[-1])
        else:
            lagged_dict, indices = segmenter.segment(hist_df[name])
            x0[name] = lagged_dict
            feat_indices[name] = indices
    for name in features_fut:  # Future features are not segmented (TODO?)
        if is_constant(fut_df, name, horizon, count_down=False):
            x0[name] = float(fut_df[name].iloc[0])
        else:
            for i in range(horizon):
                name_with_time_step = name + "_fut_" + str(i+1)
                x0[name_with_time_step] = float(fut_df[name].iloc[i])

    return x0, feat_indices



def create_masks(
    num_features: int,
    num_masks: int,
    mutation_rate: float,
    rng: random.Random,
    deterministic: bool = False,
) -> List[np.ndarray]:
    """
    Create masks for LIME perturbation generation. If deterministic,
    masks are produced as leave-one-out (plus all original).

    Args:
        num_features (int): Number of features in dataset
        num_masks (int): Number of masks to generate
        mutation_rate (float): Chance of perturbing feature, if non-deterministic
        rng (random.Random): A random number generator instance for non-deterministic generation
        deterministic (bool): Flag for whether to generate masks deterministically
    """
    if deterministic:
        masks = [np.ones(num_features, dtype=int)]
        for j in range(num_features):
            if len(masks) >= num_masks:
                break
            mask = np.ones(num_features, dtype=int)
            mask[j] = 0
            masks.append(mask)
        return masks

    return [
        np.array([1 if rng.random() >= mutation_rate else 0 for _ in range(num_features)], dtype=int)
        for _ in range(num_masks)
    ]


def perturb_vectors(
    hist_df: pd.DataFrame,
    orig_vector: Dict,
    feat_indices: Dict,
    sampler: Any,
    feature_map: List[Tuple[str, str, int | None]],
    flat_masks: List[np.ndarray],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perturb original vector to get local variations

    Args:
        hist_df (pandas DataFrame): Historical data of the specific location
        orig_vector (dict): Dictionary of original features and values
        feat_indices (dict): Dictionary of segment indices
        sampler (SampleModel): Sampler instance
        feature_map (list): Mapping of features to nested keys
        flat_masks (list): List of flat perturbation masks

    Return:
        List of perturbed vectors and list of masks
    """
    perturbations: List[Dict] = []
    perturbation_masks: List[Dict] = []

    for mask in flat_masks:
        pb: Dict = {}
        pb_mask: Dict = {}

        for idx, (_, parent_key, lag) in enumerate(feature_map):
            is_present = int(mask[idx])

            # Handle Static Features
            if lag is None:
                orig_val = orig_vector[parent_key]
                pb[parent_key] = float(orig_val) if is_present == 1 else 0.0
                pb_mask[parent_key] = is_present
                continue

            # Handle Temporal Features
            if parent_key not in pb:
                pb[parent_key] = {}
                pb_mask[parent_key] = {}

            orig_segment = orig_vector[parent_key][lag]
            if is_present == 1:
                pb[parent_key][lag] = orig_segment
            else:
                indices = feat_indices[parent_key][lag]
                pb[parent_key][lag] = sampler.sample(
                    hist_df, indices, parent_key, len(orig_segment)
                )

            pb_mask[parent_key][lag] = is_present

        perturbations.append(pb)
        perturbation_masks.append(pb_mask)

    return perturbations, perturbation_masks




def convert_vector_to_dataset(
    perturbation: Dict, 
    hist_df: pd.DataFrame, 
    fut_df: pd.DataFrame,
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    hist_type: Type,
    fut_type: Type,
    feat_indices: Dict[str, Dict]
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
        location (str): Geographical location of data,
        hist_type (type): Dataclass of historical data
        fut_type (type): Dataclass of future data
        feat_indices (Dict[str, Dict]): Dictionary of segment indices for each temporal feature
        
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
            idx_map = feat_indices[feat]
            if idx_map is None:
                raise KeyError(f"Missing indices for segmented feature '{feat}'")
    
            col_idx = hist_df.columns.get_loc(feat)

            for lag, segment_vals in val.items():
                start_idx, end_idx = idx_map[lag]
                n = end_idx - start_idx
                hist_df.iloc[start_idx:end_idx, col_idx] = segment_vals[:n]
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


def build_feature_map(
    orig_vector: Dict,
) -> List[Tuple[str, str, int | None]]:
    """
    Builds mapping from feature names to nested keys

    Args:
        orig_vector (dict): The original input vector

    Returns:
        List of tuples of feature names and their corresponding feature and lag
    """
    feature_map: List[Tuple[str, str, int | None]] = []
    for key, val in orig_vector.items():
        if isinstance(val, dict):
            for lag in val.keys():
                feature_map.append((f"{key}_lag_{lag}", key, lag))
        elif isinstance(val, (int, float)):
            feature_map.append((key, key, None))
    return feature_map


def compute_local_weights(
    weighter: Any,
    X: np.ndarray,
    x0_row: np.ndarray,
    distance_sequences: List[np.ndarray] | None = None,
    x0_sequence: np.ndarray | None = None,
) -> np.ndarray:
    if weighter.takes_mask:
        return np.asarray(weighter.get_weights(X, x0_row), dtype=float)

    return np.asarray(weighter.get_weights(distance_sequences, x0_sequence), dtype=float)


def build_distance_sequences_for_perturbations(
    perturbations: List[Dict],
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    hist_type: Type,
    fut_type: Type,
    feat_indices: Dict[str, Dict],
) -> List[np.ndarray]:
    sequences: List[np.ndarray] = []
    for pb in perturbations:
        new_hist, _ = convert_vector_to_dataset(
            pb,
            hist_df,
            fut_df,
            features_hist,
            features_fut,
            horizon,
            hist_type,
            fut_type,
            feat_indices,
        )
        seq = build_dtw_sequence(
            new_hist.to_pandas().sort_values("time_period").reset_index(drop=True),
            features_hist,
        )
        sequences.append(seq)
    return sequences


def generate_adaptive_candidate_masks(
    feature_uncertainty: np.ndarray,
    num_candidates: int,
    mutation_rate: float,
    rng: random.Random,
    blocked_masks: set[Tuple[int, ...]],
) -> List[np.ndarray]:
    """
    Generate candidate masks, biased toward features with high posterior uncertainty.

    Args:
        feature_uncertainty (numpy ndarray): Covariance matrix from surrogate
        num_candidates (int): Size of perturbation candidate pool
        mutation_rate (float): Probability of perturbing a particular feature
        rng (random.Random): Random number generator object
        blocked_masks (set): Set of already included masks not to be duplicated

    Returns:
        List of candidate masks
    """

    uncertainty = np.nan_to_num(feature_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.sum(uncertainty)) <= 0.0:
        uncertainty = np.ones(p, dtype=float)

    uncertainty = np.clip(uncertainty, 1e-12, None)
    probs = uncertainty / np.sum(uncertainty)
    # Mutation is biased towards more uncertain features
    per_feature_mutation = np.clip(mutation_rate * p * probs, 0.0, 1.0)

    candidates: List[np.ndarray] = []
    local_blocked = set(blocked_masks)

    # Include most uncertain features
    ranked = np.argsort(-uncertainty)
    for idx in ranked:
        if len(candidates) >= num_candidates:
            break
        mask = np.ones(p, dtype=int)
        mask[idx] = 0
        key = tuple(mask.tolist())
        if key in local_blocked:
            continue
        candidates.append(mask)
        local_blocked.add(key)

    # Add vectors not in dataset to pool, biased toward more uncertain features
    attempts = 0
    max_attempts = max(100, num_candidates * 25)
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        mask = np.ones(p, dtype=int)
        mutated = False

        for j in range(p):
            if rng.random() < float(per_feature_mutation[j]):
                mask[j] = 0
                mutated = True

        if not mutated:
            chosen = rng.choices(range(p), weights=probs.tolist(), k=1)[0]
            mask[chosen] = 0

        key = tuple(mask.tolist())
        if key in local_blocked:
            continue

        candidates.append(mask)
        local_blocked.add(key)

    return candidates


def build_dtw_sequence(
    hist_df: pd.DataFrame,
    features_hist: list[str],
) -> np.ndarray:
    # TODO: Only handling temporal columns... what to do with static?
    temporal_cols = [f for f in features_hist if not is_constant(hist_df, f)]
    return hist_df[temporal_cols].to_numpy(dtype=float)

def produce_lime_dataset(
    model: ExternalModel, 
    dataset: pd.DataFrame,
    future_weather: pd.DataFrame,
    perturbations: List[Dict],
    perturbation_masks: List[Dict],
    feature_names: List[str],
    features_hist: List[str],
    features_fut: List[str],
    horizon: int,
    location: str,
    feat_indices: Dict[str, List],
    hist_type: Type = None,
    fut_type: Type = None,
    chunk_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Produce training input and target dataset for LIME linear surrogate model
    
    Args:
        model (ExternalModel): Model used for producing target values
        dataset (pandas DataFrame): Dataset of historic data to perturb
        future_weather (pandas DataFrame): Dataset of future weather to perturb
        perturbations (List[Dict]): List of perturbed input vectors
        perturbation_masks (List[Dict]): List of perturbed features mapped to binary
        feature_names (List[str]): List of feature names
        features_hist (List[str]): The feature names in the historic dataset
        features_fut (List[str]): The feature names of the future dataset
        horizon (int): The number of time steps into the future for which we want detailed explanations
        location (str): Geographical location of the data
        hist_type (type): Dataclass of historical data (Default None)
        fut_type (type): Dataclass of future data (Default None)
        chunk_size (int): Size of prediction chunks
        feat_indices (Dict[str, List]): List of indices for segments, for each temporal feature

    Returns:
        Tuple of numpy arrays and feature name list
    """
    results: List[Tuple[Dict[str, float], str]] = []
    distance_sequences = []
    x0_sequence = build_dtw_sequence(dataset, features_hist)
    # Batch prediction is quicker than individual predictions
    # Key different perturbations by pseudo-location, then batch predict
    for i in range(0, len(perturbations), chunk_size):
        chunk = perturbations[i : i + chunk_size]
        chunk_masks = perturbation_masks[i : i + len(chunk)]

        full_hist_dict = {}
        full_fut_dict = {}
        pert_map = {}
        seq_map = {}

        logger.info(f"Processing prediction chunk {i//chunk_size + 1} ({len(chunk)} perturbations)...")
        for j, pb in enumerate(chunk):
            loc_id = f"pb_{i+j}"
            pert_map[loc_id] = chunk_masks[j]
            # TODO: Should refactor external_model a bit, avoid reduplication on adapt especially
            new_hist, new_fut = convert_vector_to_dataset(
                pb, 
                dataset, 
                future_weather, 
                features_hist, 
                features_fut, 
                horizon, 
                hist_type,
                fut_type,
                feat_indices
            )
            seq_map[loc_id] = build_dtw_sequence(
                new_hist.to_pandas().sort_values("time_period").reset_index(drop=True),
                features_hist,
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
                vals = avg_samples(ds)
                # Extract most recent "prob" as horizon value
                latest = max(vals.keys())
                latest_prob = vals[latest]
                flat_vec = flatten_vector(pb_vec)
                distance_sequences.append(seq_map[loc_name])
                results.append((flat_vec, latest_prob))
    
    if not results:
        raise ValueError("No results generated")

    X, y = build_X_y(results, feature_names)
    return X, y, distance_sequences, x0_sequence



def disambiguate_surrogate(name: str):
    match name.lower():
        case "ridge":
            return RidgeSurrogate()
        case "bayesian" | "blr" | "bayesian_linear":
            return BayesianSurrogate()
        case _:
            raise ValueError(f"Unknown surrogate model: {name}")


def disambiguate_segmenter(name: str, granularity: int, window_size: int | None = None): # TODO: Could maybe add a config
    match name.lower():
        case "uniform":
            return UniformSegmentation(num_segments=granularity)
        case "exponential":
            return ExponentialSegmentation(num_segments=granularity)
        case "reverse_exponential" | "reverse exponential":
            return ReverseExponentialSegmentation(num_segments=granularity)
        case "matrix_slope" | "matrix slope":
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return MatrixProfileSlopeSegmentation(num_segments=granularity, window_size=window_size)
        case "matrix_diff" | "matrix diff": # TODO: Should probably synchronize names
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return MatrixProfileSortedSlopeSegmentation(num_segments=granularity, window_size=window_size)
        case "matrix_bins" | "matrix bins":
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return MatrixProfileBinSegmentation(num_segments=granularity, window_size=window_size, num_bins=3, mode="min") # TODO: Hardcoded?
        case "sax":
            return SaxTransformSegmentation(num_segments=granularity)
        case "nn":
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return NNSegmentation(num_segments=granularity, window_size=window_size)
        case _:
            raise ValueError(f"Unknown segmenter: {name}")


def disambiguate_sampler(name: str, rng: random.Random, dataset: pd.DataFrame | None = None):
    match name.lower():
        case "background":
            if dataset is None:
                raise ValueError("Selected sampler needs dataset, which has been set to None")
            return RandomBackground(rng=rng, dataset=dataset)
        case "linear":
            return LinearInterpolation(rng=rng)
        case "constant":
            return ConstantTransform(rng=rng)
        case "local_mean" | "local mean":
            return LocalMean(rng)
        case "global_mean" | "global mean":
            return GlobalMean(rng)
        case "random":
            if dataset is None:
                raise ValueError("Selected sampler needs dataset, which has been set to None")
            return RandomUniform(rng=rng, dataset=dataset)
        case "fourier":
            return FourierReplacement(rng, dataset, window_size=None, freq=1.0)
        case _:
            raise ValueError(f"Unknown sampler: {name}")


def disambiguate_weighter(name: str, kernel_width: int):
    match name.lower():
        case "pairwise":
            return Pairwise(kernel_width)
        case "dtw":
            return DTW(kernel_width)
        case _:
            raise ValueError(f"Unknown weighter: {name}")


def print_time(start, message):
    mid = time.perf_counter()
    logger.info(message % (mid - start))


def explain(
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    granularity: int = 10,
    num_perturbations: int = 300,
    surrogate_name: str = "ridge",
    segmenter_name: str = "uniform",
    sampler_name: str = "background",
    weighter_name: str = "pairwise",
    seed: int | None = None,
    timed: bool = False
):
    """
    Model-agnostic function to supply variable contribution weighting for specific prediction
    
    Args:
        model (ExternalModel): A trained predictor on which to generate explanation
        dataset (DataSet): The dataset on which to perturb
        location (str): The location on which to explain
        horizon (int): The number of time steps into the future on which to explain
        granularity (int): Number of segments to divide the time series data into for importance weighting (default: TODO)
        num_perturbations (int): Number of generated perturbed variations of input vector (default 300)
        surrogate_name (str): The model used as explainable surrogate - one of ["ridge", "tree"] (default ridge)
        segmenter_name (str): The model used for segmentation - one of ["uniform", "exponential", "matrix_slope",
                              "matrix_diff", "matrix_bins", "sax", "nn"] (default uniform)
        sampler_name (str): The sampling strategy used to replace features "turned off" - one of ["background"] (default background)
        weighter_name (str): The strategy for weighting perturbations according to distance to original (default pairwise)
        seed (int): Seeding for RNG
        timed (bool): Flag for whether to print execution time for LIME pipeline stages
    """
    start = time.perf_counter()
    if timed:
        logger.info("Started LIME pipeline")

    # =================================================================
    # Prepare dataset
    # =================================================================

    # Initial input safety checks
    assert horizon > 0, f"Horizon must be positive; received horizon={horizon}"
    assert location in dataset.locations(), f"Location {location} not found in dataset"

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
    
    # Isolate features
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

    window_size = min(max(5, len(hist_df)//30), len(hist_df))  # TODO: Heuristic for now
    segmenter = disambiguate_segmenter(segmenter_name, granularity, window_size)


    if timed:
        print_time(start, "Finished LIME preparations in %.4f seconds")

    # =================================================================
    # Build original vector around which to generate perturbed vectors
    # =================================================================
    x0, feat_indices = build_original_vector(  # TODO: Return number of variables to scale mutation_rate accordingly
        segmenter,
        hist_df, 
        future_df, 
        features_hist, 
        features_fut, 
        horizon
    )


    if timed:
        print_time(start, "Created original LIME input vector at %.4f seconds")

    full_dataset_df = dataset.to_pandas()

    rng = random.Random(seed)
    sampler = disambiguate_sampler(sampler_name, rng, full_dataset_df)

    # =================================================================
    # Create perturbed variations
    # =================================================================
    # TODO: When, where and how to decide number of perturbations and mr?

    feature_map = build_feature_map(x0)
    feature_names = [name for name, _, _ in feature_map]

    flat_masks = create_masks(
        num_features=len(feature_names),
        num_masks=num_perturbations,
        mutation_rate=0.05,
        rng=rng,
        deterministic=False
    )

    perturbations, perturbation_masks = perturb_vectors(
        hist_df, x0, feat_indices, sampler, feature_map, flat_masks
    )

    if timed:
        print_time(start, "Finished creating perturbation vectors at %.4f seconds")

    # =================================================================
    # Get target values for generated perturbations
    # =================================================================
    
    X, y, distance_sequences, x0_sequence = produce_lime_dataset(
        model, 
        hist_df,
        future_df,
        perturbations,
        perturbation_masks,
        feature_names,
        features_hist,
        features_fut,
        horizon,
        location,
        feat_indices,
        hist_type,
        fut_type
    )

    if timed:
        print_time(start, "Finished creating LIME surrogate training dataset at %.4f seconds")
    
    x0_row = np.ones(X.shape[1], dtype=float)

    kw = 0.75 * np.sqrt(X.shape[1])  # This can be automatically inferred per paper "initial step towards stable" et.c.; magic number for now

    weighter = disambiguate_weighter(weighter_name, kw)
    if weighter.takes_mask:
        weights = weighter.get_weights(X, x0_row)
    else:
        weights = weighter.get_weights(distance_sequences, x0_sequence)
    z = np.log1p(y) # Log transform

    surrogate_model = disambiguate_surrogate(surrogate_name)
    
    if timed:
        print_time(start, "Prepared surrogate at %.4f seconds")

    # =================================================================
    # Train surrogate
    # =================================================================

    surrogate_model.fit(X, z, weights)
    
    if timed:
        print_time(start, "Trained surrogate at %.4f seconds")
    
    results = surrogate_model.explain(feature_names)

    if timed:
        print_time(start, "Finished explanation at %.4f seconds")

    # =================================================================
    # Extract explanations
    # =================================================================

    z_hat = surrogate_model.predict(X)
    r2 = r2_score(z, z_hat, sample_weight=weights)
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    logger.info(f"Surrogate weighted R2={r2:.3f}, effective N={n_eff:.1f}, p={X.shape[1]}")

    logger.info("Coefficients:")
    for name, c in results.as_sorted():
        logger.info(f"{name:>12}: {c:+.4f}")
    
    # TODO: Let user select last n datapoints to use in explanation...?
    plot_importance(results.as_sorted(), hist_df, future_df, feat_indices)


def explain_adaptive(
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    granularity: int = 10,
    num_perturbations: int = 300,
    surrogate_name: str = "ridge",
    segmenter_name: str = "uniform",
    sampler_name: str = "background",
    weighter_name: str = "pairwise",
    seed: int | None = None,
    timed: bool = False
):
    """
    Model-agnostic function using adaptive perturbation selection with a Bayesian
    linear acquisition model, then training the selected surrogate on the resulting
    local dataset.
    
    Args:
        model (ExternalModel): A trained predictor on which to generate explanation
        dataset (DataSet): The dataset on which to perturb
        location (str): The location on which to explain
        horizon (int): The number of time steps into the future on which to explain
        granularity (int): Number of segments to divide the time series data into for importance weighting (default: TODO)
        num_perturbations (int): Number of generated perturbed variations of input vector (default 300)
        surrogate_name (str): The model used as explainable surrogate - one of ["ridge"] (default ridge)
        segmenter_name (str): The model used for segmentation - one of ["uniform", "exponential", "matrix_slope",
                              "matrix_diff", "matrix_bins", "sax", "nn"] (default uniform)
        sampler_name (str): The sampling strategy used to replace features "turned off" - one of ["background"] (default background)
        weighter_name (str): The strategy for weighting perturbations according to distance to original (default pairwise)
        seed (int): Seeding for RNG
        timed (bool): Flag for whether to print execution time for LIME pipeline stages
    """
    start = time.perf_counter()
    if timed:
        logger.info("Started bayesian LIME pipeline")

    # TODO: Might be useful to break upon some evaluation stagnation, eliminating the need for num_perturbations altogether
    
    # =================================================================
    # Prepare dataset
    # =================================================================


    assert horizon > 0, f"Horizon must be positive; received horizon={horizon}"
    assert location in dataset.locations(), f"Location {location} not found in dataset"

    delta = dataset.period_range[0].time_delta
    prediction_range = PeriodRange(
        dataset.end_timestamp,
        dataset.end_timestamp + delta * horizon,
        delta,
    )

    climate_predictor = get_climate_predictor(dataset)
    future_weather = climate_predictor.predict(prediction_range)

    dataset_loc = dataset.filter_locations([location])
    future_weather = future_weather.filter_locations([location])
    hist_type = dataset_loc[location].__class__
    fut_type = future_weather[location].__class__

    hist_df = dataset_loc.to_pandas().sort_values("time_period").reset_index(drop=True)
    future_df = future_weather.to_pandas().sort_values("time_period").reset_index(drop=True)
    assert len(future_df) >= horizon, f"Need at least {horizon} future steps, got {len(future_df)}"

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

    window_size = min(max(5, len(hist_df) // 30), len(hist_df))
    segmenter = disambiguate_segmenter(segmenter_name, granularity, window_size)

    if timed:
        print_time(start, "Finished adaptive LIME preparations in %.4f seconds")

    # =================================================================
    # Build original vector around which to generate perturbed vectors
    # =================================================================

    x0, feat_indices = build_original_vector(
        segmenter,
        hist_df,
        future_df,
        features_hist,
        features_fut,
        horizon,
    )

    if timed:
        print_time(start, "Created original adaptive LIME input vector at %.4f seconds")

    full_dataset_df = dataset.to_pandas()
    rng = random.Random(seed)
    sampler = disambiguate_sampler(sampler_name, rng, full_dataset_df)

    # =================================================================
    # Create initial perturbed variations
    # =================================================================

    feature_map = build_feature_map(x0)
    feature_names = [name for name, _, _ in feature_map]
    num_features = len(feature_names)
    if num_features == 0:
        raise ValueError("No interpretable features available for explain_bayesian")

    x0_row = np.ones(num_features, dtype=float)
    x0_sequence = build_dtw_sequence(hist_df, features_hist)

    kw = 0.75 * np.sqrt(num_features)
    weighter = disambiguate_weighter(weighter_name, kw)

    mutation_rate = 0.05  # TODO: Hardcoded?
    num_initial_perturbations = min(num_perturbations, max(2, num_features + 1))
    acquisition_batch_size = max(1, num_perturbations // 10)

    initial_flat_masks = create_masks(num_features, num_initial_perturbations, mutation_rate, rng, deterministic=True)
    
    initial_perturbations, initial_perturbation_masks = perturb_vectors(
        hist_df, x0, feat_indices, sampler, feature_map, initial_flat_masks
    )
    
    # =================================================================
    # Get target values for initial generated perturbations
    # =================================================================

    X, y, distance_sequences, _ = produce_lime_dataset(
        model,
        hist_df,
        future_df,
        initial_perturbations,
        initial_perturbation_masks,
        feature_names,
        features_hist,
        features_fut,
        horizon,
        location,
        feat_indices,
        hist_type,
        fut_type,
    )

    if timed:
        print_time(start, "Finished initial adaptive perturbation evaluation at %.4f seconds")

    
    # =================================================================
    # Start adaptive pipeline
    # =================================================================

    while len(y) < num_perturbations:
        # =================================================================
        # Fit bayesian linear regression model on dataset so far
        # =================================================================
        weights = compute_local_weights(
            weighter,
            X,
            x0_row,
            distance_sequences=distance_sequences,
            x0_sequence=x0_sequence,
        )
        z = np.log1p(y)

        acquisition_surrogate = BayesianSurrogate()
        acquisition_surrogate.fit(X, z, weights)

        coef_std = acquisition_surrogate.coef_std_

        logger.info(
            f"Adaptive selection status: n={len(y)}/{num_perturbations}, "
            f"max_std={float(np.max(coef_std)):.4f}, mean_std={float(np.mean(coef_std)):.4f}"
        )

        uncertainty_tol = 0.05
        if coef_std is None:
            raise ValueError("Coef_std is None")
        if len(y) >= num_initial_perturbations and np.all(coef_std <= uncertainty_tol):  # TODO: Doesn't work per now, std must be dynamic or smt
            logger.info("Stopping adaptive sampling early because coefficient uncertainty is below threshold")
            break

        # =================================================================
        # Generate new mask candidate pool
        # =================================================================

        remaining = num_perturbations - len(y)
        batch_size = min(remaining, acquisition_batch_size)
        candidate_pool_size = max(num_features, 3 * batch_size)

        blocked_masks = {tuple(row.astype(int).tolist()) for row in X}
        candidate_flat_masks = generate_adaptive_candidate_masks(
            coef_std,
            candidate_pool_size,
            mutation_rate,
            rng,
            blocked_masks,
        )

        if not candidate_flat_masks:
            logger.info("No new candidate perturbations could be generated; stopping adaptive loop")
            break

        candidate_perturbations, candidate_perturbation_masks = perturb_vectors(
            hist_df, x0, feat_indices, sampler, feature_map, candidate_flat_masks
        )

        X_candidates = np.vstack(candidate_flat_masks).astype(float)

        # =================================================================
        # Find optimal perturbations for maximizing acquisition function 
        # =================================================================

        if weighter.takes_mask:
            candidate_weights = compute_local_weights(
                weighter,
                X_candidates,
                x0_row,
            )
        else:
            candidate_distance_sequences = build_distance_sequences_for_perturbations(
                candidate_perturbations,
                hist_df,
                future_df,
                features_hist,
                features_fut,
                horizon,
                hist_type,
                fut_type,
                feat_indices,
            )
            candidate_weights = compute_local_weights(
                weighter,
                X_candidates,
                x0_row,
                distance_sequences=candidate_distance_sequences,
                x0_sequence=x0_sequence,
            )

        scores = acquisition_surrogate.acquisition_scores(X_candidates, candidate_weights)
        top_idx = np.argsort(scores)[-batch_size:][::-1]

        selected_perturbations = [candidate_perturbations[i] for i in top_idx]
        selected_masks = [candidate_perturbation_masks[i] for i in top_idx]

        # =================================================================
        # Produce new dataset combining old and new perturbations
        # =================================================================

        X_new, y_new, distance_sequences_new, _ = produce_lime_dataset(
            model,
            hist_df,
            future_df,
            selected_perturbations,
            selected_masks,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
        )

        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        distance_sequences.extend(distance_sequences_new)

        if timed:
            mid = time.perf_counter()
            logger.info(f"Finished adaptive batch; total evaluated perturbations={len(y)} at {(mid - start):.4f} seconds")

    # =================================================================
    # Train surrogate
    # =================================================================

    weights = compute_local_weights(
        weighter,
        X,
        x0_row,
        distance_sequences=distance_sequences,
        x0_sequence=x0_sequence,
    )
    z = np.log1p(y)
    
    surrogate_model = disambiguate_surrogate(surrogate_name)

    if timed:
        print_time(start, "Prepared surrogate at %.4f seconds")

    surrogate_model.fit(X, z, weights)

    if timed:
        print_time(start, "Trained surrogate at %.4f seconds")


    # =================================================================
    # Extract explanations
    # =================================================================

    results = surrogate_model.explain(feature_names)

    if timed:
        print_time(start, "Finished explanation at %.4f seconds")

    z_hat = surrogate_model.predict(X)
    r2 = r2_score(z, z_hat, sample_weight=weights)
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    logger.info(f"Adaptive surrogate weighted R2={r2:.3f}, effective N={n_eff:.1f}, p={X.shape[1]}, n={X.shape[0]}")

    logger.info("Coefficients:")
    for name, c in results.as_sorted():
        logger.info(f"{name:>12}: {c:+.4f}")

    plot_importance(results.as_sorted(), hist_df, future_df, feat_indices)




if __name__ == "__main__":
    from pathlib import Path
    from chap_core.cli_endpoints._common import load_dataset_from_csv
    from chap_core.models.utils import get_model_from_directory_or_github_url

    model_name = "https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6"
    dataset_csv = Path("example_data/nicaragua_weekly_data.csv")
    location = "boaco"
    horizon = 3
    historical_context_steps = 6
    surrogate_name = "ridge"

    # Load dataset (without geojson handling)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path=None)

    # Load and train model directly
    estimator = get_model_from_directory_or_github_url(model_name)
    predictor = estimator.train(dataset)

    # Run explanation
    explain(  # TODO: Update
        model=predictor,
        dataset=dataset,
        location=location,
        horizon=horizon,
        granularity=historical_context_steps,
        surrogate_name=surrogate_name,
    )


# TODO: Parametrised event primitives? LOMATCE