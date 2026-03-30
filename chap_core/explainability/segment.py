"""
Functionality for segmenting time series data into interpretable blocks (segments) of data
"""

import pandas as pd
import numpy as np
import stumpy

from pyts.approximation import SymbolicAggregateApproximation
from typing import Protocol, Dict, List, Tuple



Segment = List[float]
Segments = Dict[int, Segment] 
Indices = Dict[int, Tuple[int, int]]

class SegmentationModel(Protocol):
    def segment(self, data: pd.DataFrame) -> Tuple[Segments, Indices]: ... # TODO




class UniformSegmentation(SegmentationModel):
    def __init__(self, num_segments=5):
        self.num_segments=num_segments

    def segment(self, data: pd.DataFrame) -> Tuple[Segments, Indices]:
        segments = {}
        indices = {}

        data_len = len(data)
        segment_len = data_len // self.num_segments
        
        current_idx = 0

        for i in range(self.num_segments):
            lag = (self.num_segments - 1) - i
        
            if i == self.num_segments - 1:
                subset = data.iloc[current_idx:]
            else:
                subset = data.iloc[current_idx : current_idx + segment_len]
                
            start = current_idx
            end = current_idx + len(subset)
            indices[lag] = (start, end)
            segments[lag] = subset.to_numpy().tolist()
            current_idx = end
                
        return segments, indices

class ExponentialSegmentation:
    def __init__(self, num_segments=5):
        self.num_segments=num_segments  # ExponentialSegmentation accepts num_segments but doesn't use - change?
        
    def segment(self, data: pd.DataFrame) -> Tuple[Segments, Indices]:
        segments = {}
        indices = {}

        data_len = len(data)
        d = int(np.ceil(np.log(data_len))) if data_len > 1 else 1
        current_idx = 0

        for i in range(d):
            lag = (d - 1) - i

            if i == d - 1:
                subset = data.iloc[current_idx:]
            else:
                segment_len = min(max(int(np.ceil(np.exp(i))), 1), data_len - current_idx)
                subset = data.iloc[current_idx : current_idx + segment_len]

            start = current_idx
            end = current_idx + len(subset)
            indices[lag] = (start, end)
            segments[lag] = subset.to_numpy().tolist()
            
            current_idx = end
            if current_idx >= data_len:
                break
                
        return segments, indices

class ReverseExponentialSegmentation:
    def __init__(self, num_segments=5):
        self.num_segments = num_segments  # as in ExponentialSegmentation

    def segment(self, data: pd.DataFrame) -> Tuple[Segments, Indices]:
        segments = {}
        indices = {}

        data_len = len(data)
        d = int(np.ceil(np.log(data_len))) if data_len > 1 else 1

        current_end = data_len

        for i in range(d):
            lag = i

            if i == d - 1:
                subset = data.iloc[:current_end]
                start = 0
                end = current_end
            else:
                segment_len = min(max(int(np.ceil(np.exp(i))), 1), current_end)
                start = current_end - segment_len
                end = current_end
                subset = data.iloc[start:end]

            indices[lag] = (start, end)
            segments[lag] = subset.to_numpy().tolist()

            current_end = start
            if current_end <= 0:
                break

        return segments, indices


class MatrixProfileSlopeSegmentation:
    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> Tuple[Segments, Indices]:
        segments = {}
        indices = {}

        # This ensures correct input to stumpy.stump
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("MatrixProfileSlopeSegmentation expects a Series or single-column DataFrame")
            x = data.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            # Series or array-like
            x = pd.Series(data).to_numpy(dtype=np.float64)

        data_len = len(x)
        if self.m > data_len:
            raise ValueError(f"window_size must be smaller than len(data). Got m={self.m}, data_len={data_len}")

        s = stumpy.stump(x, m=self.m)  # Create matrix profile
        mp = s[:, 0].astype(float)  # Only keep the first column, i.e. similarity distance
        grad = np.abs(np.gradient(mp))
        indexed_grad = [(i, float(v)) for i, v in enumerate(grad)]
        sorted_grad = sorted(indexed_grad, key=lambda x: x[1], reverse=True)

        k = self.num_segments - 1
        top_k = sorted_grad[:k]
        index_sorted = sorted(top_k, key=lambda x: x[0])
        boundaries = [0] + [b[0] for b in index_sorted] + [data_len]
        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = boundaries[seg_i], boundaries[seg_i + 1]
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices 


class MatrixProfileSortedSlopeSegmentation:
    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> Tuple[Segments, Indices]:
        segments: Segments = {}
        indices: Indices = {}

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("MatrixProfileSortedSlopeSegmentation expects a Series or single-column DataFrame")
            x = data.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            x = pd.Series(data).to_numpy(dtype=np.float64)

        data_len = len(x)
        if self.m > data_len:
            raise ValueError(f"window_size must be smaller than len(data). Got m={self.m}, data_len={data_len}")

        s = stumpy.stump(x, m=self.m)
        mp = s[:, 0].astype(np.float64)

        order = np.argsort(mp)
        mp_sorted = mp[order]

        jumps = np.diff(mp_sorted)

        k = self.num_segments - 1
        jump_pos = np.argsort(jumps)[-k:]
        jump_pos = np.sort(jump_pos)

        boundary_candidates = order[jump_pos + 1]
        boundary_candidates = np.sort(boundary_candidates)

        boundaries = [0] + boundary_candidates.tolist() + [data_len]

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


class MatrixProfileBinSegmentation:
    def __init__(self, num_segments: int, window_size: int, num_bins: int, mode: str = "min"):
        self.num_segments = num_segments  # As before, num_segments is not used. Remove, or concat until num_segments? TODO
        self.num_bins = num_bins
        self.m = window_size
        self.mode = mode  # one of ["min", "max"] TODO: Safety check

    def segment(self, data: pd.DataFrame | pd.Series) -> Tuple[Segments, Indices]:
        segments: Segments = {}
        indices: Indices = {}

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("MatrixProfileBinSegmentation expects a Series or single-column DataFrame")
            x = data.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            x = pd.Series(data).to_numpy(dtype=np.float64)

        data_len = len(x)

        s = stumpy.stump(x, m=self.m)
        mp = s[:, 0].astype(np.float64)
        j = len(mp)

        k = self.num_bins
        mp_min = float(np.min(mp))
        mp_max = float(np.max(mp))
        edges = np.linspace(mp_min, mp_max, k + 1)

        mp_bins = np.digitize(mp, edges[1:-1], right=False)

        base = 0 if self.mode == "min" else (k - 1)
        point_bins = np.full(data_len, base, dtype=np.int64)

        for t in range(data_len):
            a = max(0, t - self.m + 1)
            b = min(t, j - 1)
            cover = mp_bins[a : b + 1]
            point_bins[t] = int(np.min(cover)) if self.mode == "min" else int(np.max(cover))

        boundaries = [0]
        for i in range(1, data_len):
            if point_bins[i] != point_bins[i - 1]:
                boundaries.append(i)
        boundaries.append(data_len)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices



def count_runs(symbols_1d: np.ndarray) -> int:
    if len(symbols_1d) == 0:
        return 0
    runs = 1
    for i in range(1, len(symbols_1d)):
        if symbols_1d[i] != symbols_1d[i - 1]:
            runs += 1
    return runs


class SaxTransformSegmentation:
    def __init__(self, num_segments: int):
        self.num_segments = num_segments

    def segment(self, data: pd.DataFrame | pd.Series) -> Tuple[Segments, Indices]:
        # This algo doesn't really work at all on the kinds of noisy data we have
        segments: Segments = {}
        indices: Indices = {}

        if isinstance(data, pd.DataFrame):
            x = data.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            x = pd.Series(data).to_numpy(dtype=np.float64)

        n = len(x)
        target = self.num_segments
        tol = 0.10

        nan_mask = ~np.isfinite(x)
        if nan_mask.any():
            if nan_mask.all():
                # Return one single segment if all NaN
                segments[0] = x.tolist()
                indices[0] = (0, n)
                return segments, indices
            x = np.copy(x)
            x[nan_mask] = np.nanmean(x[~nan_mask])

        mu = float(np.mean(x))
        sigma = float(np.std(x))
        z = (x - mu) / (sigma if sigma != 0.0 else 1.0)

        b = 3
        max_bins = max(3, min(n, 26))  # pyts limit
        best_symbols = None

        while b <= max_bins:
            sax = SymbolicAggregateApproximation(n_bins=b, strategy="normal")
            symbols = sax.transform(z.reshape(1, -1))[0]  # Transform z to 2d because sax.transform assumes this
            runs = count_runs(symbols)

            best_symbols = symbols

            if abs(runs - target) <= max(1, int(np.ceil(target * tol))):
                break

            if runs < target:
                b += 1
            else:
                break

        boundaries = [0]
        for i in range(1, n):
            if best_symbols[i] != best_symbols[i - 1]:
                boundaries.append(i)
        boundaries.append(n)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices



def rho(x: np.ndarray, m: int, i: int, k: int) -> float:  # From the paper
    wi = x[i : i + m]
    wk = x[k : k + m]
    mu_i = float(np.mean(wi))
    mu_k = float(np.mean(wk))
    sd_i = float(np.std(wi)) or 1.0
    sd_k = float(np.std(wk)) or 1.0
    return abs((mu_i / sd_i) - (mu_k / sd_k))



class NNSegmentation:
    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> Tuple[Segments, Indices]:
        segments: Segments = {}
        indices: Indices = {}

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("NNSegmentation expects a Series or single-column DataFrame")
            x = data.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            x = pd.Series(data).to_numpy(dtype=np.float64)

        data_len = len(x)
        if self.m > data_len:
            raise ValueError(f"window_size must be smaller than len(data). Got m={self.m}, data_len={data_len}")

        s = stumpy.stump(x, m=self.m)
        nn_id = s[:, 1].astype(np.int64)  # Second column is the index of most similar section
        id_len = len(nn_id)

        candidates: List[Tuple[int, float]] = []

        for i in range(id_len - 1):
            if nn_id[i+1] != nn_id[i] + 1:
                left_i = max(0, i-self.m)
                right_i = min(id_len-1, i + self.m)
                r_left = rho(x, self.m, i, left_i)
                r_right = rho(x, self.m, i, right_i)

                if r_left > r_right:
                    candidate = i
                    score = r_left
                else:
                    candidate = i + self.m
                    score = r_right
                
                candidates.append((candidate, score))

        candidates.sort(key=lambda t: t[1], reverse=True)
        top = candidates[:self.num_segments]

        boundaries = [c for c, _ in top]
        boundaries = sorted(boundaries)

        boundaries = [0] + boundaries + [data_len]

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices




"""
Need to write documentation later TODO, but intuition is:

Uniform is equally spaced segments

Exponential is smaller segments earlier, growing exponentially, in both exponential
and uniform the last segment is widened if the data length doesn't match perfectly

Reverse exponential is same as exponential, but smaller segments last (Not found in paper)

Matrix profile slope segmentation uses matrix profile (matrix profile uses a sliding window across
and finds the most similar section elsewhere in the series to that window, and assigns this similarity as
the value of that time step) differentiated to find where sections change the most, because this is
intuitively where a unique/distinct pattern for the series would begin or end.

Matrix profile sorted slope segmentation also uses matrix profile but doesn't differentiate; instead
sorts by the largest difference in sim value between two time steps and creates boundaries at the top
k steps.

Matrix profile bin segmentation divides the time series similarities into horizontal bins, and sections of
consecutive time steps whose similarity is in the same bin becomes a single segment.

Sax transform segmentation uses the sax transform (sax transform is the division of a time series into pre-
determined bins, averaged, and binned horizontally into "symbols", where strings of these symbols form words)
to convert the time series into words, and substrings of consecutive same letters becomes one segment.
Difficult in this instance to match precisely with selected number of bins, so tolerance of 10% is given
per the paper

All transformations above, save inverse exponential, come from TS-Mule paper

NNSegment comes from the LimeSegment paper. This algo segments the series into overlapping windows of size m,
and finds the index of the most similar window. Intuitively, if we are in a region of "stability" or regularity
in values, moving along to the next time step's window would result in its most similar window being the 
window one over of the previous index of most similar. When this doesn't hold, we are probably in a regime
change and can add a candidate boundary. Select actual boundaries based on those windows which are most different
from neighbors (as function of mean and variance)

TODO: Add BEAST and others?

TODO In general: Safe getting, current development on data without NaNs which gives false security

Idea: Evaluate boundary placement by difference in mean and variance thing from above iteratively for windows
to left and right
"""