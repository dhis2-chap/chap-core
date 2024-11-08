from chap_core.time_period import PeriodRange
from chap_core.time_period.period_assignment import PeriodAssignment


def test_calculate_assignments():
    weeks = PeriodRange.from_strings(['2022W01', '2022W02', '2022W03', '2022W04', '2022W05'])
    months = PeriodRange.from_strings(['2022-01', '2022-02'])
    pa = PeriodAssignment(months, weeks)
    assert pa.indices.shape == (2, 5)
