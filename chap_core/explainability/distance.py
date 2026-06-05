"""Weighters for the LIME surrogate fit.

LIME explains a prediction by fitting a simple surrogate model to many
perturbed copies of the input. Each perturbation should count more or less
toward that fit depending on how *close* it is to the original input — the
explanation is meant to be locally faithful, so near perturbations matter
more than far ones. A weighter turns a "distance from the original" into a
locality weight via a kernel, and that weight is passed as the
``sample_weight`` of the surrogate regression.

Two strategies are provided, differing in what they measure distance over:

* :class:`Pairwise` — Euclidean distance in the binary *mask* space.
* :class:`DTW` — Dynamic Time Warping distance in the materialised
  *sequence* space.

The ``takes_mask`` flag on each weighter tells the pipeline which of those
two representations to hand to :meth:`get_weights` (see
``produce_lime_dataset`` in ``lime.py``).
"""

from typing import Protocol

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances


class WeighterModel(Protocol):
    """Contract for a LIME weighter: turn perturbations into locality weights.

    Attributes
    ----------
    takes_mask:
        Selects which representation the pipeline feeds to ``get_weights``.
        ``True`` -> the binary perturbation masks (one 0/1 vector per
        perturbation, as used by :class:`Pairwise`). ``False`` -> the
        materialised perturbed sequences (the actual feature values, as
        used by :class:`DTW`).
    """

    takes_mask: bool

    def get_weights(self, X: np.ndarray, x0_row: np.ndarray) -> list[float]: ...


class Pairwise:
    """Standard LIME weighter: Euclidean distance + Gaussian (RBF) kernel.

    This is the weighting scheme from the original LIME paper (Ribeiro et
    al., 2016). Distance is measured in the binary mask space — i.e. how
    many segments a perturbation turned off relative to the all-ones
    original — and converted to a weight with an RBF kernel
    ``exp(-d^2 / kernel_width^2)``. A perturbation identical to the
    original (distance 0) gets weight 1; the weight decays smoothly toward
    0 as more segments are turned off.

    Because it operates on masks, ``takes_mask`` is ``True``.
    """

    def __init__(self, kernel_width: int | None):
        self.kernel_width = kernel_width or 1.0
        self.takes_mask = True

    def get_weights(self, X: np.ndarray, x0_row: np.ndarray):
        """Weight each perturbation by RBF-kernelised Euclidean distance to ``x0_row``.

        Parameters
        ----------
        X:
            ``(n_perturbations, n_features)`` array of perturbation masks.
        x0_row:
            The original instance's mask (typically all ones).

        Returns
        -------
        np.ndarray
            One weight per perturbation, in ``[0, 1]``; 1 means identical
            to the original.
        """
        d = pairwise_distances(X, x0_row.reshape(1, -1), metric="euclidean").flatten()
        w = np.exp(-(d**2) / (self.kernel_width**2))
        return w


class DTW:
    """Time-series LIME weighter: Dynamic Time Warping distance + Gaussian kernel.

    Used when the perturbation should be compared as a *time series* rather
    than as a bag of independent segments. Dynamic Time Warping (DTW) aligns
    two sequences allowing local stretching/compression along the time axis,
    so it tolerates phase shifts that a point-wise Euclidean distance would
    over-penalise. The ``radius`` parameter bounds the warping window
    (FastDTW approximation; larger = closer to exact DTW, slower).

    Raw DTW distances are scaled by their batch standard deviation (for
    scale-invariance across datasets) and passed through a Gaussian kernel
    *anchored at zero distance*, so the original sequence gets the highest
    weight and weight decreases monotonically with distance. We deliberately
    do **not** mean-centre the distances first: z-normalising (as the
    reference adaptation did) would hand the highest weight to perturbations
    at the *mean* distance rather than to the original, breaking the locality
    that LIME relies on. Because it needs the actual feature values, not the
    masks, ``takes_mask`` is ``False``.
    """

    def __init__(self, kernel_width: float | None = None, radius: int = 1):
        self.kernel_width = float(kernel_width or 1.0)  # Kernel_width is scale parameter in paper
        self.radius = radius
        self.takes_mask = False

    def get_weights(
        self,
        perturbed_sequences: list[np.ndarray],
        x0_sequence: np.ndarray,
    ) -> np.ndarray:
        """Weight each perturbed sequence by sigma-scaled DTW distance to ``x0_sequence``.

        Distances are scaled by their batch standard deviation and passed
        through a Gaussian kernel anchored at zero, so distance 0 (the
        original) gets weight 1 and weight decays monotonically as distance
        grows.

        Parameters
        ----------
        perturbed_sequences:
            One materialised perturbed sequence per perturbation.
        x0_sequence:
            The original (unperturbed) sequence.

        Returns
        -------
        np.ndarray
            One weight per perturbation, in ``(0, 1]``; 1 means zero distance.
        """
        distances = np.asarray(
            [fastdtw(x0_sequence, z, radius=self.radius, dist=euclidean)[0] for z in perturbed_sequences],
            dtype=float,
        )

        sigma = distances.std()

        # Scale by sigma for scale-invariance, but keep the kernel anchored at
        # distance 0 (do NOT subtract the mean). Subtracting mu would give the
        # max weight to perturbations at the mean distance instead of to the
        # original, which undermines LIME's locality.
        scaled = distances / sigma if sigma != 0 else np.zeros_like(distances)

        weights = np.exp(-(scaled**2) / self.kernel_width)
        return np.asarray(weights)
