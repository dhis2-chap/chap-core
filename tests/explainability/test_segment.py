"""Unit tests for segmentation strategies."""

import numpy as np
import pandas as pd

from chap_core.explainability.segment import (
    ExponentialSegmentation,
    ReverseExponentialSegmentation,
    UniformSegmentation,
)


def _series(n: int = 20) -> pd.Series:
    return pd.Series(np.arange(n, dtype=float))


class TestUniformSegmentation:
    def test_exact_number_of_segments(self):
        seg = UniformSegmentation(num_segments=5)
        _, indices = seg.segment(_series(20))
        assert len(indices) == 5

    def test_indices_cover_entire_series_without_gaps(self):
        seg = UniformSegmentation(num_segments=4)
        _, indices = seg.segment(_series(16))
        # Indices are keyed by lag, not by position; reorder by start.
        ranges = sorted(indices.values())
        # First starts at 0; each segment's end is the next segment's start.
        assert ranges[0][0] == 0
        for (_, end_prev), (start_next, _) in zip(ranges, ranges[1:], strict=False):
            assert end_prev == start_next
        assert ranges[-1][1] == 16

    def test_segment_values_match_input(self):
        data = _series(10)
        seg = UniformSegmentation(num_segments=2)
        segments, indices = seg.segment(data)
        for lag, (start, end) in indices.items():
            assert segments[lag] == data.iloc[start:end].tolist()

    def test_accepts_dataframe_as_well_as_series(self):
        # The protocol widening is the whole reason this test exists.
        df = _series(12).to_frame(name="x")
        _, indices = UniformSegmentation(num_segments=3).segment(df)
        assert len(indices) == 3


class TestExponentialSegmentation:
    def test_returns_at_least_one_segment(self):
        _, indices = ExponentialSegmentation().segment(_series(8))
        assert len(indices) >= 1

    def test_indices_cover_entire_series(self):
        _, indices = ExponentialSegmentation().segment(_series(32))
        ranges = sorted(indices.values())
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 32


class TestReverseExponentialSegmentation:
    def test_indices_cover_entire_series(self):
        _, indices = ReverseExponentialSegmentation().segment(_series(32))
        ranges = sorted(indices.values())
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 32

    def test_smallest_segment_is_at_the_end(self):
        # Reverse exponential puts the finest-grained segments most recent.
        _, indices = ReverseExponentialSegmentation().segment(_series(32))
        ranges = sorted(indices.values())
        last_len = ranges[-1][1] - ranges[-1][0]
        first_len = ranges[0][1] - ranges[0][0]
        # Last (most-recent) segment shorter than first (oldest).
        assert last_len <= first_len
