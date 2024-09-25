from chap_core.data import DataSet, PeriodObservation, adaptors


class Obs(PeriodObservation):
    disease_cases: int
    rainfall: float
    temperature: float


# Create observations
observation_dict = {'Oslo': [
    Obs(time_period='2020-01', disease_cases=10, rainfall=0.1, temperature=20),
    Obs(time_period='2020-02', disease_cases=11, rainfall=0.2, temperature=22),
    Obs(time_period='2020-03', disease_cases=12, rainfall=0.3, temperature=21)],
    'Troms': [
        Obs(time_period='2020-01', disease_cases=2, rainfall=1.1, temperature=10),
        Obs(time_period='2020-02', disease_cases=2, rainfall=2.2, temperature=11),
        Obs(time_period='2020-03', disease_cases=2, rainfall=0.3, temperature=12)]}

# Create a climate health dataset
dataset = DataSet.from_period_observations(observation_dict)

# Convert to a gluonts dataset
gluonts_dataset = adaptors.gluonts.from_dataset(dataset)

print(list(gluonts_dataset))
