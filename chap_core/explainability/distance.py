"""
Functionality for calculating LIME weights based on distance between perturbed instances
"""

from typing import Protocol

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances


class WeighterModel(Protocol):
    takes_mask: bool

    def get_weights(self, X: np.ndarray, x0_row: np.ndarray) -> list[float]: ...


class Pairwise:
    def __init__(self, kernel_width: int | None):
        self.kernel_width = kernel_width or 1.0
        self.takes_mask = True

    def get_weights(self, X: np.ndarray, x0_row: np.ndarray):
        d = pairwise_distances(X, x0_row.reshape(1, -1), metric="euclidean").flatten()
        w = np.exp(-(d**2) / (self.kernel_width**2))
        return w


class DTW:
    def __init__(self, kernel_width: float | None = None, radius: int = 1):
        self.kernel_width = float(kernel_width or 1.0)  # Kernel_width is scale parameter in paper
        self.radius = radius
        self.takes_mask = False

    def get_weights(
        self,
        perturbed_sequences: list[np.ndarray],
        x0_sequence: np.ndarray,
    ) -> np.ndarray:
        distances = np.asarray(
            [fastdtw(x0_sequence, z, radius=self.radius, dist=euclidean)[0] for z in perturbed_sequences],
            dtype=float,
        )

        mu = distances.mean()
        sigma = distances.std()

        ## TODO: This transformation is what they do in the paper...
        ## but am I reading it wrong? Wouldn't perturbations close to
        ## the average (mu) be assigned a higher weight than even
        ## the original itself, since z_norm in the first case is 0
        ## and in the second case -mu/sigma?
        if sigma == 0:
            z_norm = np.zeros_like(distances)
        else:
            z_norm = (distances - mu) / sigma

        weights = np.exp(-(z_norm**2) / self.kernel_width)
        return weights
