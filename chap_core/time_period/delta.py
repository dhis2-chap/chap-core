class Delta:
    def __init__(self, n_years, n_months, n_days):
        self.n_years = n_years
        self.n_months = n_months
        self.n_days = n_days


Year = Delta(1, 0, 0)
Month = Delta(0, 1, 0)
Day = Delta(0, 0, 1)
