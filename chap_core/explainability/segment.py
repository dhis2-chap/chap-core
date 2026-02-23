"""
Functionality for segmenting time series data into interpretable blocks (segments) of data
"""
import pandas as pd
import numpy as np

from typing import Protocol, Dict, List, Tuple


Segment = List[float]
Segments = Dict[int, Segment] 
Indices = Dict[int, Tuple[int, int]]

class SegmentationModel(Protocol):
    def segment(self, data: pd.DataFrame) -> Tuple[Segments, Indices]: ... # TODO




class UniformSegmentation:
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
