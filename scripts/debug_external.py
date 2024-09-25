import chap_core.api as api


if __name__ == '__main__':
    zip_file_path = '/home/knut/Downloads/chapdata-080524-pretty (1).zip'
    out_json = 'tmp.json'
    model_name = '/home/knut/Sources/external_rmodel_example/config.yml'
    api.dhis_zip_flow(zip_file_path, out_json, model_name)
