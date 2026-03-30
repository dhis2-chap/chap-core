"""
Functionality for sampling counterfactual data for LIME
"""

import random
import pandas as pd
import numpy as np

from scipy.signal import ShortTimeFFT, get_window
from typing import Protocol, List, Tuple

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)





class SampleModel(Protocol):
    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int) -> List[float]: ...



class LinearInterpolation:
    def __init__(
            self,
            rng: random.Random,
            ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        start, end = indices
        left = max(start - 1, 0)
        right = min(len(hist_df)-1, end)
        length = (right - left) + 1

        v_left = float(hist_df[feature_name].iloc[left])
        v_right = float(hist_df[feature_name].iloc[right])
        vals = np.linspace(v_left, v_right, length)
        
        segment_vals = vals[start - left : end - left]
        return segment_vals.tolist()

        
class ConstantTransform:
    def __init__(
            self,
            rng: random.Random,
            ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        segment = [0.0]*length
        return segment


class LocalMean:
    def __init__(
            self,
            rng: random.Random,
            ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        start, end = indices
        mean = float(np.mean(hist_df[feature_name].iloc[start:end]))
        return [mean] * length

class GlobalMean:
    def __init__(
            self,
            rng: random.Random,
            ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        mean = float(np.mean(hist_df[feature_name]))
        return [mean] * length

class RandomUniform:
    def __init__(
            self,
            rng: random.Random,
            dataset: pd.DataFrame
            ):
        self.rng = rng
        self.dataset = dataset

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        max_val = np.max(self.dataset[feature_name])
        min_val = np.min(self.dataset[feature_name])
        rand_vals = [self.rng.uniform(min_val, max_val) for i in range(length)]
        return rand_vals


class RandomBackground:
    def __init__(
            self,
            rng: random.Random,
            dataset: pd.DataFrame,
            ):
        self.rng = rng
        self.dataset = dataset

    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        locations = self.dataset["location"].dropna().unique().tolist()

        for _ in range(3): # Retries twice if selected window is too narrow
            loc = self.rng.choice(locations)
            loc_df: pd.DataFrame = self.dataset[self.dataset["location"] == loc]

            if len(loc_df) < length:
                continue

            max_start = len(loc_df) - length            
            start = self.rng.randrange(0, max_start + 1)
            vals = loc_df.iloc[start : start + length][feature_name].astype(float).tolist()

            if any(pd.isna(v) for v in vals):
                continue

            return vals
        
        # Fallback
        logger.info("Background sampling failed; fallback to constant value")
        return [0.0] * length


class FourierReplacement:
    def __init__(
            self,
            rng: random.Random,
            dataset: pd.DataFrame,
            window_size: int,
            freq: float
            ):
        self.rng = rng
        self.dataset = dataset
        self.window_size = window_size
        self.freq = freq

        self.R_cache = {}

    def calculate_background(self, ts: np.ndarray):
        # Safety guards to avoid NaN errors
        # TODO: Other method than replace with nanmean?
        if np.any(~np.isfinite(ts)):
            clean = np.copy(ts)
            mask = ~np.isfinite(clean)
            if mask.all():
                logger.warning("Time series is all NaN")
                return np.zeros_like(ts)
            clean[mask] = np.nanmean(clean[~mask])
            ts = clean

        ts_length = len(ts)
        if self.window_size is None:
            dynamic_win = ts_length // 10
            win_size = min(max(16, dynamic_win), ts_length - 1)
        else:
            win_size = self.window_size

        win_size = max(2, min(win_size, ts_length - 1))
        
        win = get_window('hann', win_size)
        stft = ShortTimeFFT(win, hop=1, fs=self.freq)
        x_stft = stft.stft(ts)
        f_persist = self.extract_argmax(x_stft)

        f_filtered = np.zeros_like(x_stft)
        f_filtered[f_persist, :] = x_stft[f_persist, :]
        R = stft.istft(f_filtered, k0=0, k1=len(ts))

        if np.any(~np.isfinite(R)):
            logger.warning("Fourier background contains NaN/inf; falling back to original series")
            R = np.copy(ts)

        return R
    
    def extract_argmax(self, x_stft: np.ndarray):
        mag = np.abs(x_stft)
        mean_mag = np.mean(mag, axis=1)
        std_mag = np.std(mag, axis=1)
        score = mean_mag / (std_mag + 1e-9)
        return np.argmax(score)
    
    def sample(self, hist_df: pd.DataFrame, indices: Tuple[int, int], feature_name: str, length: int):
        start, end = indices
        if feature_name not in self.R_cache:
            ts = hist_df[feature_name].to_numpy()
            self.R_cache[feature_name] = self.calculate_background(ts)
        
        segment = np.asarray(self.R_cache[feature_name][start:end], dtype=float)

        if len(segment) != length or not np.isfinite(segment).all():
            logger.warning(
                f"Invalid Fourier segment for {feature_name}; falling back to original segment"
            )
            segment = hist_df[feature_name].iloc[start:end].to_numpy(dtype=float)

        if len(segment) != length or not np.isfinite(segment).all():
            logger.warning(
                f"Original segment for {feature_name} is also invalid; using zeros"
            )
            segment = np.zeros(length, dtype=float)
        return segment.tolist()



"""
Linear transform, constant transform, random background is from "agnostic local explanations [...]"
Local mean, global mean, random uniform is from "LOMATCE"
Fourier is from LimeSegment

"""

# TODO: cprofile