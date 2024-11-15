from chap_core.api_types import RequestV1
from chap_core.rest_api_src.worker_functions import dataset_from_request_v1

json_data = RequestV1.model_validate_json(open('/home/knut/Data/ch_data/chap_request_data_togo.json', 'r').read())
if __name__ == '__main__':
    dataset = dataset_from_request_v1(json_data, target_name='disease_cases', usecwd_for_credentials=True)