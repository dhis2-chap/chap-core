
def extract_last(samples):
    last_sample = {key: value[-1] if not hasattr(value, 'items') else extract_last(value) for key, value in samples.items()}
    if 'log_observation_rate' in last_sample:
        print('-------------------------------')
        print(last_sample['log_observation_rate'])
        print(samples['log_observation_rate'])
    return last_sample