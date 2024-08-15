from climate_health.api import train_with_validation

#train_with_validation('FlaxModel', 'laos_full_data')
train_with_validation('ProbabilisticFlaxModel', 'laos_full_data')