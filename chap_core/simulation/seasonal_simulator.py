import numpy as np


class SeasonalSingleVariableSimulator:
    """Simulates a single variable with seasonal effect"""

    def __init__(
        self,
        n_seasons: int,
        n_data_points_per_season: int,
        mean_peak_height: int,
        peak_height_sd: float,
    ):
        self.n_seasons = n_seasons
        self.n_data_points_per_season = n_data_points_per_season
        self.mean_peak_height = mean_peak_height
        self.peak_height_sd = peak_height_sd
        self.data_size = self.n_seasons * self.n_data_points_per_season

    def simulate_peak_positions(self):
        bin_size = self.data_size // self.n_seasons
        peak_positions = np.zeros(self.n_seasons, dtype=int)
        for i in range(self.n_seasons):
            peak_positions[i] = np.random.randint(low=(i * bin_size) + 1, high=((i + 1) * bin_size) - 1)
        return peak_positions

    def simulate_peak_heights(self):
        return np.random.normal(self.mean_peak_height, self.peak_height_sd, self.n_seasons)

    def simulate_valley_positions(self, peak_positions: np.ndarray):
        valley_positions = np.zeros(self.n_seasons + 1, dtype=int)
        valley_positions[0] = np.random.choice(peak_positions[0])
        for i in range(1, self.n_seasons):
            valley_positions[i] = np.random.randint(low=peak_positions[i - 1] + 1, high=peak_positions[i] - 1)
        valley_positions[-1] = np.random.randint(low=peak_positions[-1] + 1, high=self.data_size)
        return valley_positions

    def simulate_valley_heights(self, peak_heights: np.ndarray):
        valley_heights = np.zeros(self.n_seasons + 1, dtype=int)
        valley_means = peak_heights * 0.3
        valley_means = np.insert(valley_means, 0, peak_heights[0] * 0.3)
        valley_heights = np.random.normal(valley_means, valley_means * 0.1)
        return valley_heights

        """
        valley_heights[0] = np.random.choice(peak_heights[0])
        for i in range(1, self.n_seasons):
            weighted_choices = [np.random.randint(low=1, high=int(peak_heights[i-1]*0.1)+2), np.random.randint(low=1, high=peak_heights[i-1]+1)]
            valley_heights[i] = np.random.choice(weighted_choices, p=[0.8, 0.2])
        return valley_heights
        """

    def simulate(self):
        data = np.zeros(self.data_size, dtype=int)
        peak_positions = self.simulate_peak_positions()
        peak_heights = self.simulate_peak_heights()
        valley_positions = self.simulate_valley_positions(peak_positions)
        valley_heights = self.simulate_valley_heights(peak_heights)
        data[peak_positions] = peak_heights
        data[valley_positions] = valley_heights
        nonzero_indices = np.nonzero(data)
        for i in range(len(nonzero_indices[0]) - 1):
            start = nonzero_indices[0][i]
            end = nonzero_indices[0][i + 1]
            data[start:end] = np.linspace(data[start], data[end], end - start, dtype=int)
        return data
