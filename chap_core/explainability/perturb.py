"""
Functionality for sampling counterfactual data for LIME
"""

import random
import pandas as pd
import numpy as np

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


# TODO: FFT-based algo from LimeSegment


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



