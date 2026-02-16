"""
Functionality for segmenting time series data into interpretable blocks (segments) of data
"""
import pandas as pd

from typing import Protocol, Dict, List


Segment = List[float]
Segments = Dict[int, Segment] 

class SegmentationModel(Protocol):
    def segment(self, data: pd.DataFrame) -> Segments: ... # TODO




class UniformSegmentation:
    def __init__(self, num_segments=5):
        self.num_segments=num_segments

    def segment(self, data: pd.DataFrame) -> Segments:
        segments = {}
        data_len = len(data)
        segment_len = data_len // self.num_segments
        
        current_idx = 0

        for i in range(self.num_segments):
            lag = (self.num_segments - 1) - i
        
            if i == self.num_segments - 1:
                    subset = data.iloc[current_idx:]
            else:
                subset = data.iloc[current_idx : current_idx + segment_len]
                current_idx += segment_len
        
            segments[lag] = subset.values.flatten().tolist()
        
        return segments