import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from chap_core.climate_predictor import get_climate_predictor
from chap_core.exceptions import ModelFailedException
from chap_core.explainability.distance import *
from chap_core.explainability.perturb import *
from chap_core.explainability.plot import plot_importance
from chap_core.explainability.segment import *
from chap_core.explainability.surrogate import *
from chap_core.model_spec import _non_feature_names
from chap_core.models.external_model import ExternalModel
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import PeriodRange

logger = logging.getLogger(__name__)

_non_feature_names_plus_disease = _non_feature_names - {"disease_cases"}


def avg_samples(
    data: DataSet,
) -> dict:
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
    if sample_cols.empty:
        raise ValueError("No sample columns found in prediction output")
    mean_vals = dataframe[sample_cols].mean(axis=1)
    tp_len = len(dataframe["time_period"])
    val_len = len(mean_vals)
    assert tp_len == val_len, f"Error: Length of time period {tp_len} is not equal to length of values {val_len}"
    return dict(zip(dataframe["time_period"], mean_vals, strict=False))


def is_constant(window: pd.DataFrame, feature_name: str, num_steps: int | None = None, from_end: bool = True) -> bool:
    """
    Check whether a feature varies with time.

    Args:
        window (pandas.DataFrame): Dataframe containing the time series.
        feature_name (str): Name of the column to check.
        num_steps (int | None): If given, only inspect this many rows. When
            from_end is True the tail is checked; otherwise the head.
        from_end (bool): When num_steps is set, True checks the last
            num_steps rows; False checks the first.

    Returns:
        True if the feature takes at most one distinct value in the checked range.
    """
    series = window[feature_name]

    if num_steps is not None:
        series = series.iloc[-num_steps:] if from_end else series.iloc[:num_steps]

    return series.nunique(dropna=False) <= 1


def build_original_vector(
    segmenter: SegmentationModel,
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    features_hist: list[str],
    features_fut: list[str],
    horizon: int,
) -> tuple[dict, dict[str, Indices]]:
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

    # TODO (future work): Handle categorical

    # Segment historical data
    for name in features_hist:
        if is_constant(hist_df, name):  # If features doesn't vary with time, add once
            x0[name] = float(hist_df[name].iloc[-1])
        else:
            # lagged_dict is a dictionary over "lag indices" with the actual data in those segments
            # indices is a dictionary over the lag indices with the start and end indices
            lagged_dict, indices = segmenter.segment(hist_df[name])
            x0[name] = lagged_dict
            feat_indices[name] = indices

    for name in features_fut:  # Future features are not segmented since these are almost always few in number
        if is_constant(fut_df, name, horizon, from_end=False):
            x0[name] = float(fut_df[name].iloc[0])
        else:
            for i in range(horizon):
                name_with_time_step = name + "_fut_" + str(i + 1)
                x0[name_with_time_step] = float(fut_df[name].iloc[i])

    return x0, feat_indices


def create_masks(
    num_features: int,
    num_masks: int,
    mutation_rate: float,
    rng: random.Random,
    deterministic: bool = False,
) -> list[np.ndarray]:
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
        masks = [np.ones(num_features, dtype=int)]  # Include original vector among masks
        for j in range(num_features):
            if len(masks) >= num_masks:
                break
            mask = np.ones(num_features, dtype=int)
            mask[j] = 0
            masks.append(mask)
        return masks

    masks = [np.ones(num_features, dtype=int)]
    while len(masks) < num_masks:
        mask = np.array([1 if rng.random() >= mutation_rate else 0 for _ in range(num_features)], dtype=int)
        if mask.sum() == num_features:
            # Force at least one perturbation
            mask[rng.randrange(num_features)] = 0
        masks.append(mask)
    return masks


def perturb_vectors(
    hist_df: pd.DataFrame,
    orig_vector: dict,
    feat_indices: dict,
    sampler: SampleModel,
    feature_map: list[tuple[str, str, int | None]],
    flat_masks: list[np.ndarray],
    global_means: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Perturb original vector to get local variations

    Args:
        hist_df (pandas DataFrame): Historical data of the specific location
        orig_vector (dict): Dictionary of original features and values
        feat_indices (dict): Dictionary of segment indices
        sampler (SampleModel): Sampler instance
        feature_map (list): Mapping of features to nested keys
        flat_masks (list): List of flat perturbation masks
        global_means (dict | None): Global mean per feature across all locations,
            used as the replacement value for static features when perturbed.
            If None (single-location dataset), static features fall back to 0.0.

    Return:
        List of perturbed vectors and list of masks
    """
    perturbations: list[dict] = []
    perturbation_masks: list[dict] = []

    for mask in flat_masks:  # TODO: Currently all features are perturbed equally?
        pb: dict = {}
        pb_mask: dict = {}

        for idx, (_, parent_key, lag) in enumerate(feature_map):
            is_present = int(mask[idx])

            # Handle Static Features
            if lag is None:
                orig_val = orig_vector[parent_key]
                if is_present == 1:
                    pb[parent_key] = float(orig_val)
                else:
                    # Use global mean across all locations as the turned off feature,
                    # or 0.0 when the full dataset contains only one location. TODO 0.0 could be meaningless OOD
                    pb[parent_key] = global_means.get(parent_key, 0.0) if global_means else 0.0
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
                pb[parent_key][lag] = sampler.sample(hist_df, indices, parent_key, len(orig_segment))

            pb_mask[parent_key][lag] = is_present

        perturbations.append(pb)
        perturbation_masks.append(pb_mask)

    # Perturbations is the actual perturbed data, perturbation masks is the nested structure of masks
    return perturbations, perturbation_masks


def convert_vector_to_dataset(
    perturbation: dict,
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    features_hist: list[str],
    features_fut: list[str],
    horizon: int,
    hist_type: type,
    fut_type: type,
    feat_indices: dict[str, dict],
) -> tuple[DataSet, DataSet]:
    """
    Convert interpretable vector back into full perturbed dataset

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

    for feat in features_hist:
        if feat not in perturbation:
            continue

        val = perturbation[feat]
        if isinstance(val, dict):  # i.e. if value is temporal/non-static
            idx_map = feat_indices[feat]
            if idx_map is None:
                raise KeyError(f"Missing indices for segmented feature '{feat}'")

            col_idx = hist_df.columns.get_loc(feat)

            for lag, segment_vals in val.items():
                start_idx, end_idx = idx_map[lag]
                n = end_idx - start_idx
                col_vals = hist_df[feat].dropna().values
                arr = np.asarray(segment_vals[:n], dtype=float)

                if not np.isfinite(arr).all():
                    raise ValueError(f"Non-finite perturbed values for feature {feat}: {arr}")
                # Avoid casting to float if feature is int-like
                # Originally only cross-checked with np.integer but this still caused errors;
                # comparing roundedness is a catch-all check
                is_int_like = np.issubdtype(hist_df[feat].dtype, np.integer) or (
                    len(col_vals) > 0 and np.allclose(col_vals, np.round(col_vals))
                )
                if is_int_like:
                    arr = np.clip(arr, 0, None)
                    arr = np.round(arr).astype(int)

                # Insert perturbed section
                hist_df.iloc[start_idx:end_idx, col_idx] = arr
        else:
            hist_df[feat] = float(val)

    # Future data insertions
    for feat in features_fut:
        has_steps = any(f"{feat}_fut_{i + 1}" in perturbation for i in range(horizon))

        if has_steps:
            for i in range(horizon):
                key = f"{feat}_fut_{i + 1}"
                if key in perturbation and feat in fut_df.columns:
                    fut_df.loc[i, feat] = float(perturbation[key])
        else:
            if feat in perturbation and feat in fut_df.columns:
                fut_df.loc[: horizon - 1, feat] = float(perturbation[feat])

    new_hist = DataSet.from_pandas(hist_df, dataclass=hist_type)
    new_fut = DataSet.from_pandas(fut_df, dataclass=fut_type)

    return new_hist, new_fut


def build_X_y(results: list[tuple[dict, float | int]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
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


def flatten_vector(vector: dict) -> dict[str, int]:
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
    orig_vector: dict,
) -> list[tuple[str, str, int | None]]:
    """
    Builds mapping from nested keys to feature names

    Args:
        orig_vector (dict): The original input vector

    Returns:
        List of tuples of feature names and their corresponding feature and lag
    """
    feature_map: list[tuple[str, str, int | None]] = []
    # Loop over original vector and extract key (feature name) and lag index
    for key, val in orig_vector.items():
        if isinstance(val, dict):
            feature_map.extend((f"{key}_lag_{lag}", key, lag) for lag in val)
        elif isinstance(val, (int, float)):
            feature_map.append((key, key, None))

    # feature_map is a list of lagged feature names (e.g. temperature_lag_3)
    # and corresponding feature name and lag (e.g. temperature and 3)
    return feature_map


def compute_local_weights(
    weighter,
    X: np.ndarray,
    x0_row: np.ndarray,
    distance_sequences: list[np.ndarray] | None = None,
    x0_sequence: np.ndarray | None = None,
) -> np.ndarray:
    if weighter.takes_mask:
        return np.asarray(weighter.get_weights(X, x0_row), dtype=float)

    return np.asarray(weighter.get_weights(distance_sequences, x0_sequence), dtype=float)


def build_distance_sequences_for_perturbations(
    perturbations: list[dict],
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    features_hist: list[str],
    features_fut: list[str],
    horizon: int,
    hist_type: type,
    fut_type: type,
    feat_indices: dict[str, dict],
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
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
    blocked_masks: set[tuple[int, ...]],
) -> list[np.ndarray]:
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
    p = len(uncertainty)
    if float(np.sum(uncertainty)) <= 0.0:
        uncertainty = np.ones(p, dtype=float)

    uncertainty = np.clip(uncertainty, 1e-12, None)
    probs = uncertainty / np.sum(uncertainty)
    # Mutation is biased towards more uncertain features
    per_feature_mutation = np.clip(mutation_rate * p * probs, 0.0, 1.0)

    candidates: list[np.ndarray] = []
    local_blocked = set(blocked_masks)  # TODO (future work): Some samplers are non-deterministic

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
    seq = hist_df[temporal_cols].to_numpy(dtype=float)
    # NaN-check
    mask = ~np.isfinite(seq)
    if mask.any():
        df_tmp = (
            pd.DataFrame(seq).ffill().bfill().fillna(0.0)
        )  # Fill NaN forwards, fallback to backwards, fallback to 0
        seq = df_tmp.to_numpy(dtype=float)
    return seq


def produce_lime_dataset(
    model: ExternalModel,
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    perturbations: list[dict],
    perturbation_masks: list[dict],
    feature_names: list[str],
    features_hist: list[str],
    features_fut: list[str],
    horizon: int,
    location: str,
    feat_indices: dict[str, list],
    hist_type: type | None = None,
    fut_type: type | None = None,
    chunk_size: int = 10,
    full_dataset: DataSet | None = None,
    full_future_weather: DataSet | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Produce training input and target dataset for LIME linear surrogate model

    Args:
        model (ExternalModel): Model used for producing target values
        hist_df (pandas DataFrame): Dataset of historic data to perturb
        future_df (pandas DataFrame): Dataset of future weather to perturb
        perturbations (List[Dict]): List of perturbed input vectors
        perturbation_masks (List[Dict]): List of perturbed features mapped to binary
        feature_names (List[str]): List of feature names
        features_hist (List[str]): The feature names in the historic dataset
        features_fut (List[str]): The feature names of the future dataset
        horizon (int): The number of time steps into the future for which we want detailed explanations
        location (str): Geographical location of the data
        feat_indices (Dict[str, List]): List of indices for segments, for each temporal feature
        hist_type (type): Dataclass of historical data (Default None)
        fut_type (type): Dataclass of future data (Default None)
        chunk_size (int): Size of prediction chunks

    Returns:
        Tuple of input and output arrays, perturbed and original dtw distance sequence/s
    """
    results: list[tuple[dict[str, float], float]] = []
    distance_sequences = []
    x0_sequence = build_dtw_sequence(hist_df, features_hist)

    try:
        # Batch prediction by pseudo-location
        for i in range(0, len(perturbations), chunk_size):
            chunk = perturbations[i : i + chunk_size]
            chunk_masks = perturbation_masks[i : i + len(chunk)]

            full_hist_dict = {}
            full_fut_dict = {}
            pert_map = {}
            seq_map = {}

            logger.info(f"Processing prediction chunk {i // chunk_size + 1} ({len(chunk)} perturbations)...")
            for j, pb in enumerate(chunk):
                # Predict multiple outputs at once by assigning input data to different "locations" in same df
                loc_id = f"pb_{i + j}"
                pert_map[loc_id] = chunk_masks[j]
                new_hist, new_fut = convert_vector_to_dataset(
                    pb, hist_df, future_df, features_hist, features_fut, horizon, hist_type, fut_type, feat_indices
                )
                # Also store dtw sequence in case dtw distancing is used
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

    except ModelFailedException:  # If the model one-hot-encodes location, input dimension would not match training data when location is singled out
        if full_dataset is None or full_future_weather is None:
            raise
        logger.info("Batch predict failed; retrying perturbations individually with full dataset")
        results = []
        distance_sequences = []
        for j, (pb, pb_mask) in enumerate(zip(perturbations, perturbation_masks, strict=False)):
            logger.info(f"Processing perturbation {j + 1} / {len(perturbations)}...")
            new_hist, new_fut = convert_vector_to_dataset(
                pb, hist_df, future_df, features_hist, features_fut, horizon, hist_type, fut_type, feat_indices
            )
            seq = build_dtw_sequence(
                new_hist.to_pandas().sort_values("time_period").reset_index(drop=True),
                features_hist,
            )
            hist_dict = {loc: full_dataset[loc] for loc in full_dataset.locations()}
            hist_dict[location] = new_hist.get_location(location)
            fut_dict = {loc: full_future_weather[loc] for loc in full_future_weather.locations()}
            fut_dict[location] = new_fut.get_location(location)
            pred_v = model.predict(DataSet(hist_dict, polygons=None), DataSet(fut_dict, polygons=None))
            vals = avg_samples(pred_v.filter_locations([location]))
            latest = max(vals.keys())
            distance_sequences.append(seq)
            results.append((flatten_vector(pb_mask), vals[latest]))

    if not results:
        raise ValueError("No results generated")

    # X is here the perturbation masks, y the black box model output
    X, y = build_X_y(results, feature_names)
    return X, y, distance_sequences, x0_sequence


def disambiguate_surrogate(name: str) -> SurrogateModel:
    match name.lower():
        case "ridge":
            return RidgeSurrogate()
        case "bayesian" | "blr" | "bayesian_linear":
            return BayesianSurrogate()
        case _:
            raise ValueError(f"Unknown surrogate model: {name}")


def disambiguate_segmenter(name: str, granularity: int, window_size: int | None = None) -> SegmentationModel:
    """
    Fetch the actual segmenter instance from the short name

    Args:
        name (str): Short name of segmenter to use
        granularity (int): Number of segments for the segmenter to produce
        window_size (int | None, optional): Size of sliding window for certain segmenters. Defaults to None.

    Returns:
        SegmentationModel: Returns the segmentation model with a method "segment"
    """

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
        case "matrix_diff" | "matrix diff":  # TODO: Should probably synchronize names
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return MatrixProfileSortedSlopeSegmentation(num_segments=granularity, window_size=window_size)
        case "matrix_bins" | "matrix bins":
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return MatrixProfileBinSegmentation(
                num_segments=granularity, window_size=window_size, num_bins=3, mode="min"
            )  # TODO: Hardcoded?
        case "sax":
            return SaxTransformSegmentation(num_segments=granularity)
        case "nn":
            if window_size is None:
                raise ValueError("Selected segmenter needs a window_size, which is currently set to None")
            return NNSegmentation(num_segments=granularity, window_size=window_size)
        case _:
            raise ValueError(f"Unknown segmenter: {name}")


def disambiguate_sampler(name: str, rng: random.Random, dataset: pd.DataFrame | None = None) -> SampleModel:
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


def save_explanation(
    results: list[tuple[str, float]],
    model_name: str | None,
    location: str,
    horizon: int,
    r2: float,
    n_eff: float,
    params: dict,
) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (model_name or "unknown"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = CHAP_RUNS_DIR / "explainability" / safe_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LIME Explanation",
        "",
        f"**Model:** {model_name or 'unknown'}  ",
        f"**Location:** {location}  ",
        f"**Horizon:** {horizon}  ",
        f"**Generated:** {timestamp}  ",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in params.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Surrogate Quality",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Weighted R² | {r2:.3f} |",
        f"| Effective N | {n_eff:.1f} |",
        f"| Features | {len(results)} |",
        "",
        "## Feature Importance",
        "",
        "| Feature | Coefficient |",
        "|---------|-------------|",
    ]
    for name, coef in results:
        lines.append(f"| {name} | {coef:+.4f} |")

    md_path = run_dir / "explanation.md"
    md_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Explanation saved to {md_path}")
    return md_path


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
    last_n: int | None = None,
    seed: int | None = None,
    timed: bool = False,
    save: bool = True,
    plot: bool = True,
    return_metrics: bool = False,
):
    """
    Model-agnostic function to supply variable contribution weighting for specific prediction

    Args:
        model (ExternalModel): A trained predictor on which to generate explanation
        dataset (DataSet): The dataset on which to perturb
        location (str): The location on which to explain
        horizon (int): The number of time steps into the future on which to explain
        granularity (int): Number of segments to divide the time series data into for importance weighting (default: 10)
        num_perturbations (int): Number of generated perturbed variations of input vector (default 300)
        surrogate_name (str): The model used as explainable surrogate - one of ["ridge", "tree"] (default ridge)
        segmenter_name (str): The model used for segmentation - one of ["uniform", "exponential", "matrix_slope",
                              "matrix_diff", "matrix_bins", "sax", "nn"] (default uniform)
        sampler_name (str): The sampling strategy used to replace features "turned off" - one of ["background"] (default background)
        weighter_name (str): The strategy for weighting perturbations according to distance to original (default pairwise)
        last_n (int | None): If set, only the last last_n time steps of the location's historical data are used for the explanation.
        seed (int): Seeding for RNG
        timed (bool): Flag for whether to print execution time for LIME pipeline stages
        save (bool): Whether to save the calculated importance weighting
        plot (bool): Whether to plot the calculated importance weighting
        return_metrics (bool): If True, also compute and return evaluation metrics alongside the explanation
    """
    start = time.perf_counter()
    if timed:
        logger.info("Started LIME pipeline")

    # =================================================================
    # Prepare dataset
    # TODO: The setup block below is duplicated in explain_adaptive
    # =================================================================

    # Initial input safety checks
    assert horizon > 0, f"Horizon must be positive; received horizon={horizon}"
    assert location in dataset.locations(), f"Location {location} not found in dataset"

    # Determine length of time for which to predict into the future (given by horizon)
    delta = dataset.period_range[0].time_delta
    prediction_range = PeriodRange(
        dataset.end_timestamp,
        dataset.end_timestamp + delta * horizon,
        delta,
    )

    # Make future prediction for climate columns
    climate_data = dataset
    for field_name in dataset.field_names():
        if getattr(next(iter(dataset.values())), field_name).dtype.kind not in (
            "f",
            "i",
        ):  # Remove any non-numeric field from climate predictor
            climate_data = climate_data.remove_field(field_name)
    climate_predictor = get_climate_predictor(climate_data)
    full_future_weather = climate_predictor.predict(prediction_range)

    # Isolate dataset to selected location
    dataset_loc = dataset.filter_locations([location])
    future_weather = full_future_weather.filter_locations([location])

    # Fetch dataframe class to use in later instantiation
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
        if fn not in _non_feature_names_plus_disease and hist_df[fn].dtype.kind in ("f", "i")
    ]
    features_fut = [
        fn
        for fn in future_weather.field_names()
        if fn not in _non_feature_names_plus_disease and future_df[fn].dtype.kind in ("f", "i")
    ]

    assert len(features_hist) > 0, "No numeric historical features found in dataset"

    # Optionally restrict explanation to the most recent time steps
    hist_df = hist_df.copy()
    if last_n is not None:
        assert last_n > 0, f"last_n must be positive, got {last_n}"
        hist_df = hist_df.iloc[-last_n:].reset_index(drop=True)
        assert len(hist_df) > 0, f"No data remaining after selecting last {last_n} steps"

    # Handle any missing values
    nan_counts = hist_df[features_hist].isna().sum()
    if nan_counts.any():
        logger.warning(
            "Missing values detected in historical features: %s. Applying forward/backward fill.",
            nan_counts[nan_counts > 0].to_dict(),
        )
    hist_df[features_hist] = hist_df[features_hist].ffill().bfill()

    # Preparation for time series segmentation
    # Window size is used in matrix profiling for some segmenters, as the size of a sliding window
    window_size = min(max(5, len(hist_df) // 30), len(hist_df))  # TODO: Heuristic for now
    segmenter = disambiguate_segmenter(segmenter_name, granularity, window_size)

    if timed:
        print_time(start, "Finished LIME preparations in %.4f seconds")

    # =================================================================
    # Build original vector around which to generate perturbed vectors
    # =================================================================

    # Build the input vector for the prediction to explain
    x0, feat_indices = build_original_vector(segmenter, hist_df, future_df, features_hist, features_fut, horizon)

    if timed:
        print_time(start, "Created original LIME input vector at %.4f seconds")

    full_dataset_df = dataset.to_pandas()

    # Select sampler from sampler_name
    rng = random.Random(seed)
    sampler = disambiguate_sampler(sampler_name, rng, full_dataset_df)

    num_locations = len(list(dataset.locations()))
    global_means: dict[str, float] | None = (
        {feat: float(full_dataset_df[feat].mean()) for feat in features_hist} if num_locations > 1 else None
    )

    # =================================================================
    # Create perturbed variations
    # =================================================================

    # Get structured list of new feature names (e.g. rainfall_lag_5) and their column names/lag indices
    feature_map = build_feature_map(x0)
    feature_names = [name for name, _, _ in feature_map]

    # Scale mutation rate so that on average exactly one feature is perturbed per mask
    mutation_rate = 1.0 / len(feature_names)

    # Create perturbation masks - these masks decide which features will be perturbed for a given vector
    # Mask [1, 1, 0, 1] would perturb the second to last feature
    flat_masks = create_masks(
        num_features=len(feature_names),
        num_masks=num_perturbations,
        mutation_rate=mutation_rate,
        rng=rng,
        deterministic=False,  # In standard LIME, perturbation is a stochastic process
    )

    # Create the actual perturbations - these are variations on the original input vector where
    # some of the segments in the time series have been altered
    perturbations, perturbation_masks = perturb_vectors(
        hist_df, x0, feat_indices, sampler, feature_map, flat_masks, global_means=global_means
    )

    if timed:
        print_time(start, "Finished creating perturbation vectors at %.4f seconds")

    # =================================================================
    # Get target values for generated perturbations
    # =================================================================

    # The surrogate model must be trained on the masks and the output of the original model
    # on the perturbed data, obtained here. Also obtain dtw sequences (simply the temporal columns in
    # a dataframe) for perturbations and original input, for later potential dtw distancing
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
        fut_type,
        full_dataset=dataset,
        full_future_weather=full_future_weather,
    )

    if timed:
        print_time(start, "Finished creating LIME surrogate training dataset at %.4f seconds")

    x0_row = np.ones(X.shape[1], dtype=float)

    # 3/4 * sqrt(num_features) is the most common kernel width in LIME papers
    # TODO: Can be automatically inferred per paper "initial step towards stable" et.c.
    kw = 0.75 * np.sqrt(X.shape[1])

    weighter = disambiguate_weighter(weighter_name, kw)
    weights = compute_local_weights(weighter, X, x0_row, distance_sequences, x0_sequence)

    # Log transform for output with potentially long tail
    z = np.log1p(y)

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

    # Temporary metrics TODO: update
    z_hat = surrogate_model.predict(X)
    r2 = r2_score(z, z_hat, sample_weight=weights)  # Compare similarity of output in neighborhood of prediction
    n_eff = (weights.sum() ** 2) / (
        weights**2
    ).sum()  # Effective number of perturbations, based on the distribution of weighting of perturbations
    logger.info(f"Surrogate weighted R2={r2:.3f}, effective N={n_eff:.1f}, p={X.shape[1]}")

    sorted_results = results.as_sorted()

    logger.info("Coefficients:")
    for name, c in sorted_results:
        logger.info(f"{name:>12}: {c:+.4f}")

    # =================================================================
    # Calculate metrics
    # =================================================================

    metrics = {"r2": r2, "n_eff": float(n_eff)}

    if return_metrics:
        from chap_core.explainability.testing.metrics import eLoss

        mask_type1 = np.ones(X.shape[1])
        pb_orig, pb_mask_orig = perturb_vectors(
            hist_df, x0, feat_indices, sampler, feature_map, [mask_type1], global_means=global_means
        )
        _, y_orig_arr, _, _ = produce_lime_dataset(
            model,
            hist_df,
            future_df,
            pb_orig,
            pb_mask_orig,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
            full_dataset=dataset,
            full_future_weather=full_future_weather,
        )
        y_orig = y_orig_arr[0]

        delta_eloss, auc_t1, auc_t2 = eLoss(
            model=model,
            original_vector=x0,
            feature_map=feature_map,
            sorted_explanation=sorted_results,
            sampler=sampler,
            hist_df=hist_df,
            fut_df=future_df,
            feature_names=feature_names,
            features_hist=features_hist,
            features_fut=features_fut,
            horizon=horizon,
            location=location,
            hist_type=hist_type,
            fut_type=fut_type,
            feat_indices=feat_indices,
            y_orig=y_orig,
            full_dataset=dataset,
            full_future_weather=full_future_weather,
        )

        logger.info(f"EVALUATION: Delta eLoss = {delta_eloss:.4f} (Type1 AUC: {auc_t1:.4f}, Type2 AUC: {auc_t2:.4f})")
        metrics["delta_eloss"] = delta_eloss
        metrics["auc_type1"] = auc_t1
        metrics["auc_type2"] = auc_t2

    # =================================================================
    # Plot and save
    # =================================================================

    if plot:
        plot_importance(sorted_results, hist_df, future_df, feat_indices)

    if save:
        save_explanation(
            results=sorted_results,
            model_name=model.name,
            location=location,
            horizon=horizon,
            r2=r2,
            n_eff=float(n_eff),
            params={
                "segmenter": segmenter_name,
                "sampler": sampler_name,
                "surrogate": surrogate_name,
                "weighter": weighter_name,
                "granularity": granularity,
                "num_perturbations": num_perturbations,
                "seed": seed,
                "adaptive": False,
            },
        )

    if return_metrics:
        return sorted_results, metrics
    return sorted_results


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
    last_n: int | None = None,
    seed: int | None = None,
    timed: bool = False,
    save: bool = True,
    plot: bool = True,
    return_metrics: bool = False,
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
        granularity (int): Number of segments to divide the time series data into for importance weighting (default: 10)
        num_perturbations (int): Number of generated perturbed variations of input vector (default 300)
        surrogate_name (str): The model used as explainable surrogate - one of ["ridge"] (default ridge)
        segmenter_name (str): The model used for segmentation - one of ["uniform", "exponential", "matrix_slope",
                              "matrix_diff", "matrix_bins", "sax", "nn"] (default uniform)
        sampler_name (str): The sampling strategy used to replace features "turned off" - one of ["background"] (default background)
        weighter_name (str): The strategy for weighting perturbations according to distance to original (default pairwise)
        last_n (int | None): If set, only the last last_n time steps of the location's historical data are used for the explanation.
        seed (int): Seeding for RNG
        timed (bool): Flag for whether to print execution time for LIME pipeline stages
        save (bool): Whether to save the calculated importance weighting
        plot (bool): Whether to plot the calculated importance weighting
        return_metrics (bool): If True, also compute and return evaluation metrics alongside the explanation
    """
    start = time.perf_counter()
    if timed:
        logger.info("Started bayesian LIME pipeline")

    # =================================================================
    # Prepare dataset
    # TODO: The setup block below is duplicated in explain
    # =================================================================

    assert horizon > 0, f"Horizon must be positive; received horizon={horizon}"
    assert location in dataset.locations(), f"Location {location} not found in dataset"

    delta = dataset.period_range[0].time_delta
    prediction_range = PeriodRange(
        dataset.end_timestamp,
        dataset.end_timestamp + delta * horizon,
        delta,
    )

    climate_data = dataset
    for field_name in dataset.field_names():
        if getattr(next(iter(dataset.values())), field_name).dtype.kind not in ("f", "i"):
            climate_data = climate_data.remove_field(field_name)
    climate_predictor = get_climate_predictor(climate_data)
    full_future_weather = climate_predictor.predict(prediction_range)

    dataset_loc = dataset.filter_locations([location])
    future_weather = full_future_weather.filter_locations([location])
    hist_type = dataset_loc[location].__class__
    fut_type = future_weather[location].__class__

    hist_df = dataset_loc.to_pandas().sort_values("time_period").reset_index(drop=True)
    future_df = future_weather.to_pandas().sort_values("time_period").reset_index(drop=True)

    assert len(future_df) >= horizon, f"Need at least {horizon} future steps, got {len(future_df)}"

    features_hist = [
        fn
        for fn in dataset_loc.field_names()
        if fn not in _non_feature_names_plus_disease and hist_df[fn].dtype.kind in ("f", "i")
    ]
    features_fut = [
        fn
        for fn in future_weather.field_names()
        if fn not in _non_feature_names_plus_disease and future_df[fn].dtype.kind in ("f", "i")
    ]

    assert len(features_hist) > 0, "No numeric historical features found in dataset"

    # Optionally restrict explanation to the most recent time steps
    hist_df = hist_df.copy()
    if last_n is not None:
        assert last_n > 0, f"last_n must be positive, got {last_n}"
        hist_df = hist_df.iloc[-last_n:].reset_index(drop=True)
        assert len(hist_df) > 0, f"No data remaining after selecting last {last_n} steps"

    # Handle any missing values
    nan_counts = hist_df[features_hist].isna().sum()
    if nan_counts.any():
        logger.warning(
            "Missing values detected in historical features: %s. Applying forward/backward fill.",
            nan_counts[nan_counts > 0].to_dict(),
        )
    hist_df[features_hist] = hist_df[features_hist].ffill().bfill()

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

    num_locations = len(list(dataset.locations()))
    global_means: dict[str, float] | None = (
        {feat: float(full_dataset_df[feat].mean()) for feat in features_hist} if num_locations > 1 else None
    )

    # =================================================================
    # Create initial perturbed variations
    # =================================================================

    feature_map = build_feature_map(x0)
    feature_names = [name for name, _, _ in feature_map]
    num_features = len(feature_names)
    if num_features == 0:
        raise ValueError("No interpretable features available for explain_adaptive")

    x0_row = np.ones(num_features, dtype=float)
    x0_sequence = build_dtw_sequence(hist_df, features_hist)

    kw = 0.75 * np.sqrt(num_features)
    weighter = disambiguate_weighter(weighter_name, kw)

    mutation_rate = 1.0 / num_features
    num_initial_perturbations = min(num_perturbations, max(2, num_features + 1))
    acquisition_batch_size = max(1, num_perturbations // 10)

    initial_flat_masks = create_masks(num_features, num_initial_perturbations, mutation_rate, rng, deterministic=True)

    initial_perturbations, initial_perturbation_masks = perturb_vectors(
        hist_df, x0, feat_indices, sampler, feature_map, initial_flat_masks, global_means=global_means
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
        full_dataset=dataset,
        full_future_weather=full_future_weather,
    )

    if timed:
        print_time(start, "Finished initial adaptive perturbation evaluation at %.4f seconds")

    # =================================================================
    # Start adaptive pipeline
    # =================================================================

    # Stop early if maximum coefficient uncertainty does not improve by at least
    # STAGNATION_TOL for STAGNATION_PATIENCE consecutive iterations
    prev_max_std = float("inf")
    stagnation_count = 0
    STAGNATION_PATIENCE = 3
    STAGNATION_TOL = 1e-3

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

        if coef_std is None:
            raise ValueError("Coef_std is None")

        current_max_std = float(np.max(coef_std))
        logger.info(
            f"Adaptive selection status: n={len(y)}/{num_perturbations}, "
            f"max_std={current_max_std:.4f}, mean_std={float(np.mean(coef_std)):.4f}"
        )

        uncertainty_tol = 0.05
        if len(y) >= num_initial_perturbations and np.all(coef_std <= uncertainty_tol):
            logger.info("Stopping adaptive sampling early: coefficient uncertainty below threshold")
            break

        if prev_max_std - current_max_std < STAGNATION_TOL:
            stagnation_count += 1
            if stagnation_count >= STAGNATION_PATIENCE:
                logger.info("Stopping adaptive sampling: coefficient uncertainty not improving")
                break
        else:
            stagnation_count = 0
        prev_max_std = current_max_std

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
            hist_df, x0, feat_indices, sampler, feature_map, candidate_flat_masks, global_means=global_means
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
            full_dataset=dataset,
            full_future_weather=full_future_weather,
        )

        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        distance_sequences.extend(distance_sequences_new)

        if timed:
            mid = time.perf_counter()
            logger.info(
                f"Finished adaptive batch; total evaluated perturbations={len(y)} at {(mid - start):.4f} seconds"
            )

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

    sorted_results = results.as_sorted()

    logger.info("Coefficients:")
    for name, c in sorted_results:
        logger.info(f"{name:>12}: {c:+.4f}")

    # =================================================================
    # Calculate metrics
    # =================================================================

    z_hat = surrogate_model.predict(X)
    r2 = r2_score(z, z_hat, sample_weight=weights)
    n_eff = (weights.sum() ** 2) / (weights**2).sum()
    logger.info(f"Adaptive surrogate weighted R2={r2:.3f}, effective N={n_eff:.1f}, p={X.shape[1]}, n={X.shape[0]}")

    metrics = {"r2": r2, "n_eff": float(n_eff)}

    if return_metrics:
        from chap_core.explainability.testing.metrics import eLoss

        mask_type1 = np.ones(X.shape[1])
        pb_orig, pb_mask_orig = perturb_vectors(
            hist_df, x0, feat_indices, sampler, feature_map, [mask_type1], global_means=global_means
        )
        _, y_orig_arr, _, _ = produce_lime_dataset(
            model,
            hist_df,
            future_df,
            pb_orig,
            pb_mask_orig,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
            full_dataset=dataset,
            full_future_weather=full_future_weather,
        )
        y_orig = y_orig_arr[0]

        delta_eloss, auc_t1, auc_t2 = eLoss(
            model=model,
            original_vector=x0,
            feature_map=feature_map,
            sorted_explanation=sorted_results,
            sampler=sampler,
            hist_df=hist_df,
            fut_df=future_df,
            feature_names=feature_names,
            features_hist=features_hist,
            features_fut=features_fut,
            horizon=horizon,
            location=location,
            hist_type=hist_type,
            fut_type=fut_type,
            feat_indices=feat_indices,
            y_orig=y_orig,
            full_dataset=dataset,
            full_future_weather=full_future_weather,
        )

        logger.info(f"EVALUATION: Delta eLoss = {delta_eloss:.4f} (Type1 AUC: {auc_t1:.4f}, Type2 AUC: {auc_t2:.4f})")
        metrics["delta_eloss"] = delta_eloss
        metrics["auc_type1"] = auc_t1
        metrics["auc_type2"] = auc_t2

    # =================================================================
    # Plot and save
    # =================================================================

    if plot:
        plot_importance(sorted_results, hist_df, future_df, feat_indices)

    if save:
        save_explanation(
            results=sorted_results,
            model_name=model.name,
            location=location,
            horizon=horizon,
            r2=r2,
            n_eff=float(n_eff),
            params={
                "segmenter": segmenter_name,
                "sampler": sampler_name,
                "surrogate": surrogate_name,
                "weighter": weighter_name,
                "granularity": granularity,
                "num_perturbations": num_perturbations,
                "seed": seed,
                "adaptive": True,
            },
        )

    if return_metrics:
        return sorted_results, metrics
    return sorted_results
