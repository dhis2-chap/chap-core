import numpy as np

from chap_core.time_period import PeriodRange
from chap_core.time_period.date_util_wrapper import delta_day


class PeriodAssignment:
    '''Matches two period ranges with possibly different time deltas and gives assignments to each period in the first range.'''

    def __init__(self, to_range: PeriodRange, from_range: PeriodRange):
        self.to_range = to_range
        self.from_range = from_range
        self._from_range_days = from_range.delta//delta_day
        assignments = self._calculate_assignments()
        self.indices, self.weights = assignments

    def _calculate_assignments(self):
        '''Calculate the assignments for each period in the first range'''
        assignments = []
        for period in self.to_range:
            matches = self._match_period(period)
            print(matches)
            assignments.append(matches)
        max_len = max(len(matches) for matches in assignments)
        indices= np.zeros((len(assignments), max_len),
                          dtype=int)
        weights = np.zeros((len(assignments), max_len),
                           dtype=float)
        for i, matches in enumerate(assignments):
            for j, (index, weight) in enumerate(matches):
                indices[i, j] = index
                weights[i, j] = weight
        return indices, weights

    def _match_period(self, to_period):
        matches = []
        for i, from_period in enumerate(self.from_range):
            max_start = max(to_period.start_timestamp, from_period.start_timestamp)
            min_end = min(to_period.end_timestamp, from_period.end_timestamp)
            overlap = max((min_end - max_start)//delta_day, 0)
            if overlap > 0:
                matches.append((i, overlap/self._from_range_days))
        return matches

