import json

import pandas as pd

from chap_core.api_types import EvaluationResponse

json_filename = 'evaluation_response_uganda.json'
evaluation_response = EvaluationResponse.model_validate(json.load(open(json_filename)))
df = pd.read_csv('/home/knut/Data/ch_data/uganda_weekly_data.csv')
name_mapping = {row['organisationunitid'].lower(): row['organisationunitname'] for _, row in df.iterrows()}

for entry in evaluation_response.predictions:
    if entry.orgUnit.lower() in name_mapping:
        entry.orgUnit = name_mapping[entry.orgUnit.lower()]
for data_element in evaluation_response.actualCases.data:
    if data_element.ou.lower() in name_mapping:
        data_element.ou = name_mapping[data_element.ou.lower()]

serialized_response = evaluation_response.json()
out_filename = 'evaluation_response_human_uganda.json'
with open(out_filename, 'w') as out_file:
    out_file.write(serialized_response)