"""Segmenters for time-series LIME.

A time series has one value per time step, which is far too fine-grained to
perturb directly — you'd need an astronomical number of perturbations to
cover every value. Segmentation groups consecutive time steps into a handful
of contiguous blocks ("segments"), and LIME then perturbs a whole segment at
a time. Each segment becomes one interpretable feature in the explanation.

Every segmenter returns two parallel dicts keyed by **lag** (lag 0 = the most
recent segment, higher lag = older):

* ``Segments`` — ``{lag: [values...]}``, the raw values in each segment.
* ``Indices`` — ``{lag: (start_row, end_row)}``, so a perturbed segment can
  be written back into the right rows of the dataframe.

Strategy provenance: every segmenter except `ReverseExponentialSegmentation`
(uniform, exponential, the three matrix-profile variants, SAX) comes from the
TS-Mule paper; `NNSegmentation` comes from LimeSegment;
`ReverseExponentialSegmentation` is a chap-core addition not from any paper.
"""

from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd
import stumpy
from pyts.approximation import SymbolicAggregateApproximation

Segment = list[float]
Segments = dict[int, Segment]
Indices = dict[int, tuple[int, int]]


def _finalize_boundaries(internal_candidates: Iterable[int], data_len: int) -> list[int]:
    """Turn raw internal boundary candidates into a clean ascending boundary list.

    Drops anything outside the open interval ``(0, data_len)``, dedupes, sorts,
    and brackets the result with ``0`` and ``data_len``. This guarantees the
    consecutive pairs never produce a zero-length segment — candidate selection
    in the matrix-profile / nearest-neighbour segmenters can otherwise emit a
    boundary at 0, at ``data_len``, or a duplicate, all of which yield empty
    ``(x, x)`` segments that become meaningless lag features.
    """
    clean = sorted({int(b) for b in internal_candidates if 0 < int(b) < data_len})
    return [0, *clean, data_len]


class SegmentationModel(Protocol):
    """Contract for a segmenter: split a single feature series into lag-keyed segments.

    ``segment`` takes a one-column DataFrame or a Series and returns
    ``(segments, indices)`` — the per-lag values and their ``(start, end)``
    row spans (lag 0 = most recent).
    """

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]: ...


class UniformSegmentation(SegmentationModel):
    """Equally-spaced segments.

    Cuts the series into ``num_segments`` blocks of equal length; the last
    (most recent, lag 0) segment absorbs any remainder when the length
    doesn't divide evenly. The simplest strategy and a sensible default.
    """

    def __init__(self, num_segments=5):
        self.num_segments = num_segments

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
        segments: Segments = {}
        indices: Indices = {}

        data_len = len(data)
        # Cap the segment count at the data length: asking for more segments than
        # rows would otherwise produce empty (start == end) segments that become
        # meaningless lag features and waste perturbation budget.
        num_segments = min(self.num_segments, data_len)
        if num_segments == 0:
            return segments, indices
        segment_len = data_len // num_segments

        current_idx = 0

        for i in range(num_segments):
            lag = (num_segments - 1) - i

            if i == num_segments - 1:
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
    """Exponentially-growing segments, finest at the recent end.

    Segment lengths grow as ``ceil(exp(i))`` so the oldest data is lumped
    into a few wide blocks while recent data is split finely — useful when
    recent time steps carry more explanatory weight. The number of segments
    is derived from the data length (``ceil(log(n))``), so ``num_segments``
    is currently accepted but unused. The last block absorbs the remainder.
    """

    def __init__(self, num_segments=5):
        self.num_segments = num_segments  # ExponentialSegmentation accepts num_segments but doesn't use - change?

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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
    """Exponentially-growing segments, finest at the *oldest* end.

    Mirror image of :class:`ExponentialSegmentation`: the smallest segments
    are the most recent (lag 0) and they grow toward the past. Not from any
    paper — a chap-core addition for when older data is the finer-grained
    region of interest.
    """

    def __init__(self, num_segments=5):
        self.num_segments = num_segments  # as in ExponentialSegmentation

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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
    """Boundaries where the matrix profile changes fastest.

    The matrix profile slides a length-``window_size`` window across the
    series and, for each position, records the distance to its most similar
    window elsewhere. Differentiating that profile (``|gradient|``) highlights
    where the series' local pattern changes most sharply — intuitively where
    one regime ends and another begins. Boundaries are placed at the top
    ``num_segments - 1`` gradient peaks.
    """

    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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
        boundaries = _finalize_boundaries((b[0] for b in top_k), data_len)
        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = boundaries[seg_i], boundaries[seg_i + 1]
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


class MatrixProfileSortedSlopeSegmentation:
    """Boundaries at the largest jumps in the *sorted* matrix profile.

    Like :class:`MatrixProfileSlopeSegmentation` but instead of
    differentiating the profile in time order, it sorts the profile values
    and finds the ``num_segments - 1`` largest jumps between consecutive
    sorted values; the original positions of those jumps become boundaries.
    Separates the series by similarity-value level rather than by where
    changes happen in time.
    """

    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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

        boundaries = _finalize_boundaries(boundary_candidates.tolist(), data_len)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


class MatrixProfileBinSegmentation:
    """Boundaries where the matrix profile crosses into a different quantile bin.

    Divides the matrix-profile values into ``num_bins`` horizontal bins, then
    assigns each time step the min (or max, per ``mode``) bin of the windows
    covering it. Runs of consecutive time steps in the same bin become one
    segment — grouping the series by similarity *level* rather than by change
    points. ``num_segments`` is currently unused (the bin structure
    determines the count).
    """

    def __init__(self, num_segments: int, window_size: int, num_bins: int, mode: str = "min"):
        self.num_segments = (
            num_segments  # As before, num_segments is not used. Remove, or concat until num_segments? TODO
        )
        self.num_bins = num_bins
        self.m = window_size
        self.mode = mode  # one of ["min", "max"] TODO: Safety check

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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
        boundaries.extend(i for i in range(1, data_len) if point_bins[i] != point_bins[i - 1])
        boundaries.append(data_len)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


def count_runs(symbols_1d: np.ndarray) -> int:
    """Count maximal runs of identical consecutive symbols (used to size SAX segments)."""
    if len(symbols_1d) == 0:
        return 0
    runs = 1
    for i in range(1, len(symbols_1d)):
        if symbols_1d[i] != symbols_1d[i - 1]:
            runs += 1
    return runs


class SaxTransformSegmentation:
    """Boundaries between runs of the same SAX symbol.

    Symbolic Aggregate Approximation (SAX) z-normalises the series and bins
    each value into a discrete symbol; runs of the same symbol form a
    segment. The number of bins is searched upward (from 3) until the run
    count is within 10% of ``num_segments`` (tolerance per the paper). Note:
    SAX struggles on the noisy data this project typically sees.
    """

    def __init__(self, num_segments: int):
        self.num_segments = num_segments

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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
        best_symbols: np.ndarray | None = None

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

        # max_bins is always >= 3 and b starts at 3, so the loop runs at least
        # once and best_symbols is always assigned. Assert for the type checker.
        assert best_symbols is not None
        boundaries = [0]
        boundaries.extend(i for i in range(1, n) if best_symbols[i] != best_symbols[i - 1])
        boundaries.append(n)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


def rho(x: np.ndarray, m: int, i: int, k: int) -> float:  # From the paper
    """Dissimilarity between two length-`m` windows (at `i` and `k`) by mean/std ratio.

    Used by :class:`NNSegmentation` to score candidate boundaries.
    """
    wi = x[i : i + m]
    wk = x[k : k + m]
    mu_i = float(np.mean(wi))
    mu_k = float(np.mean(wk))
    sd_i = float(np.std(wi)) or 1.0
    sd_k = float(np.std(wk)) or 1.0
    return abs((mu_i / sd_i) - (mu_k / sd_k))


class NNSegmentation:
    """Boundaries at nearest-neighbour discontinuities (LimeSegment).

    Slides overlapping length-``window_size`` windows and, via the matrix
    profile, finds each window's nearest-neighbour index. In a stable regime
    the NN index advances by one as you step forward; when it jumps, you're
    likely at a regime change. Those discontinuities are candidate
    boundaries, scored by how dissimilar the neighbouring windows are (see
    :func:`rho`); the top ``num_segments`` become actual boundaries.
    """

    def __init__(self, num_segments: int, window_size: int):
        self.num_segments = num_segments
        self.m = window_size

    def segment(self, data: pd.DataFrame | pd.Series) -> tuple[Segments, Indices]:
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

        candidates: list[tuple[int, float]] = []

        for i in range(id_len - 1):
            if nn_id[i + 1] != nn_id[i] + 1:
                left_i = max(0, i - self.m)
                right_i = min(id_len - 1, i + self.m)
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
        top = candidates[: self.num_segments]

        boundaries = _finalize_boundaries((c for c, _ in top), data_len)

        num_segs = len(boundaries) - 1
        for seg_i in range(num_segs):
            start, end = int(boundaries[seg_i]), int(boundaries[seg_i + 1])
            lag = (num_segs - 1) - seg_i
            indices[lag] = (start, end)
            segments[lag] = x[start:end].tolist()

        return segments, indices


# Per-strategy intuition now lives in each class's docstring above.
#
# TODO: Add BEAST and other segmenters?
# TODO (general): harden NaN handling — current development is on data without
#   NaNs, which gives a false sense of security.
# Idea: evaluate boundary placement by iterating the rho() mean/variance
#   dissimilarity over windows to the left and right of each candidate.
