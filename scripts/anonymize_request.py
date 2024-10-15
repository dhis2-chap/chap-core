import numpy as np

from chap_core.api_types import RequestV2

filepath = '/home/knut/Data/ch_data/chap_request.json'
data: RequestV2 = RequestV2.model_validate_json(open(filepath, 'r').read())
for features in data.features:
    for entry in features.data:
        entry.value = np.random.randint(1000, 2000)

new_file_path = '../example_data/anonymous_chap_request.json'
with open(new_file_path, 'w') as f:
    f.write(data.model_dump_json())

