"""Perturbation samplers for LIME.

When a LIME perturbation "turns off" a segment of a time-series feature, the
pipeline has to put *something* back in its place. Replacing it with zero is
rarely right — zero rainfall is a real signal, not the absence of one — so
each sampler here implements a different notion of a "neutral" replacement
for a segment.

All samplers share the :class:`SampleModel` contract: given the history for
one location, the ``(start, end)`` row indices of the segment to replace,
the feature column name, and how many values are needed, return that many
replacement values.

The strategies come from three different time-series LIME papers:

* :class:`LinearInterpolation`, :class:`ConstantTransform`,
  :class:`RandomBackground` — "agnostic local explanations [...]".
* :class:`LocalMean`, :class:`GlobalMean`, :class:`RandomUniform` — LOMATCE.
* :class:`FourierReplacement` — LimeSegment.
"""

import logging
import random
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.signal import ShortTimeFFT, get_window

logger = logging.getLogger(__name__)


class SampleModel(Protocol):
    """Contract for a perturbation sampler: produce replacement values for one "off" segment.

    Parameters of :meth:`sample`
    ----------------------------
    hist_df:
        Historical data for the single location being explained.
    indices:
        ``(start, end)`` row positions of the segment to replace.
    feature_name:
        Column in ``hist_df`` the segment belongs to.
    length:
        Number of replacement values to return (normally ``end - start``).
    """

    def sample(
        self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int
    ) -> list[float]: ...


class LinearInterpolation:
    """Replace the segment with a straight line between its neighbouring boundary values.

    Takes the value just before the segment and just after it and fills the
    gap with ``np.linspace`` between them — a smooth ramp that ignores
    whatever was actually inside. Falls back to zeros if either boundary is
    non-finite. From "agnostic local explanations [...]".
    """

    def __init__(
        self,
        rng: random.Random,
    ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        start, end = indices
        left = max(start - 1, 0)
        right = min(len(hist_df) - 1, end)
        length = (right - left) + 1

        v_left = float(hist_df[feature_name].iloc[left])
        v_right = float(hist_df[feature_name].iloc[right])

        if not np.isfinite(v_left) or not np.isfinite(v_right):
            logger.warning("NaN boundary value for feature '%s'; falling back to zeros", feature_name)
            return [0.0] * (end - start)

        vals = np.linspace(v_left, v_right, length)

        segment_vals = vals[start - left : end - left]
        return segment_vals.tolist()


class ConstantTransform:
    """Replace the segment with zeros.

    The cheapest baseline and the most semantically loaded — zero is itself
    a meaningful value for most covariates, so this rarely gives a truly
    "neutral" perturbation. From "agnostic local explanations [...]".
    """

    def __init__(
        self,
        rng: random.Random,
    ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        segment = [0.0] * length
        return segment


class LocalMean:
    """Replace the segment with the mean of the segment itself, repeated.

    Flattens the segment to its own average — removes the within-segment
    variation while keeping its overall level. Falls back to zeros if the
    segment is all-NaN. From LOMATCE.
    """

    def __init__(
        self,
        rng: random.Random,
    ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        start, end = indices
        segment = hist_df[feature_name].iloc[start:end]
        if segment.isna().all():
            logger.warning("All-NaN local segment for feature '%s'; falling back to zeros", feature_name)
            return [0.0] * length
        return [float(np.nanmean(segment))] * length


class GlobalMean:
    """Replace the segment with the mean of the whole feature series, repeated.

    Like :class:`LocalMean` but the average is taken over the entire history
    of the feature, not just the segment — neutralises the segment toward
    the feature's global baseline. Falls back to zeros if the series is
    all-NaN. From LOMATCE.
    """

    def __init__(
        self,
        rng: random.Random,
    ):
        self.rng = rng

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        series = hist_df[feature_name]
        if series.isna().all():
            logger.warning("All-NaN global series for feature '%s'; falling back to zeros", feature_name)
            return [0.0] * length
        return [float(np.nanmean(series))] * length


class RandomUniform:
    """Replace each value in the segment with an independent uniform draw from the feature's range.

    Draws from ``[min, max]`` of the feature *across the whole dataset*, so
    the replacement is plausible in magnitude but carries no temporal
    structure. The most aggressive sampler — most likely to push the model
    out of distribution. From LOMATCE.
    """

    def __init__(self, rng: random.Random, dataset: pd.DataFrame):
        self.rng = rng
        self.dataset = dataset

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        max_val = np.max(self.dataset[feature_name])
        min_val = np.min(self.dataset[feature_name])
        rand_vals = [self.rng.uniform(min_val, max_val) for i in range(length)]
        return rand_vals


class RandomBackground:
    """Replace the segment with a real contiguous window sampled from another location.

    Picks a random location in the dataset and copies a same-length window
    of the feature's real values from it. Keeps realistic temporal structure
    (unlike :class:`RandomUniform`) while decoupling it from the explained
    location's history. Retries a couple of times if the chosen location's
    window is too short or contains NaNs, then falls back to zeros. From
    "agnostic local explanations [...]".
    """

    def __init__(
        self,
        rng: random.Random,
        dataset: pd.DataFrame,
    ):
        self.rng = rng
        self.dataset = dataset

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        locations = self.dataset["location"].dropna().unique().tolist()

        if not locations:
            logger.warning(
                "Background dataset has no non-null locations for feature '%s'; falling back to zeros", feature_name
            )
            return [0.0] * length

        for _ in range(3):  # Retries twice if selected window is too narrow
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
    """Replace the segment with a frequency-filtered ("background") version of the series.

    The LimeSegment strategy: run a Short-Time Fourier Transform over the
    whole feature series, keep only the single most *persistent* frequency
    band (high mean magnitude relative to its variance), and invert back to
    the time domain. The reconstruction ``R`` is a smooth seasonal/background
    signal with the transient detail stripped out; the segment is replaced
    with the matching slice of ``R``. Per-feature reconstructions are cached
    in ``R_cache``. Falls back to the original segment, then to zeros, if the
    reconstruction is invalid. From LimeSegment.
    """

    def __init__(
        self,
        rng: random.Random,
        dataset: pd.DataFrame | None,
        window_size: int | None,
        freq: float,
    ):
        self.rng = rng
        self.dataset = dataset
        self.window_size = window_size
        self.freq = freq

        self.R_cache: dict[str, np.ndarray] = {}

    def calculate_background(self, ts: np.ndarray):
        """Compute the frequency-filtered background reconstruction for a full time series.

        Cleans non-finite values, picks an STFT window size (dynamic if
        ``window_size`` is None), keeps only the most persistent frequency
        band (see :meth:`extract_argmax`) and inverts the STFT. Returns a
        same-length array; zeros if the series is all-NaN.
        """
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

        win = get_window("hann", win_size)
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
        """Return the index of the most persistent frequency band in an STFT.

        Scores each frequency by mean magnitude divided by its std across
        time (high, steady energy = persistent), and returns the argmax.
        """
        mag = np.abs(x_stft)
        mean_mag = np.mean(mag, axis=1)
        std_mag = np.std(mag, axis=1)
        score = mean_mag / (std_mag + 1e-9)
        return np.argmax(score)

    def sample(self, hist_df: pd.DataFrame, indices: tuple[int, int], feature_name: str, length: int):
        start, end = indices
        if feature_name not in self.R_cache:
            ts = hist_df[feature_name].to_numpy()
            self.R_cache[feature_name] = self.calculate_background(ts)

        segment = np.asarray(self.R_cache[feature_name][start:end], dtype=float)

        if len(segment) != length or not np.isfinite(segment).all():
            logger.warning(f"Invalid Fourier segment for {feature_name}; falling back to original segment")
            segment = hist_df[feature_name].iloc[start:end].to_numpy(dtype=float)

        if len(segment) != length or not np.isfinite(segment).all():
            logger.warning(f"Original segment for {feature_name} is also invalid; using zeros")
            segment = np.zeros(length, dtype=float)
        return segment.tolist()
