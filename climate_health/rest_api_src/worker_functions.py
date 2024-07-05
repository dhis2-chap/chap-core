from climate_health.api import read_zip_folder, train_on_prediction_data


def train_on_zip_file(file, model_name, model_path, control=None):
    print('train_on_zip_file')
    print('F', file)
    prediction_data = read_zip_folder(file.file)

    return train_on_prediction_data(prediction_data, model_name=model_name, model_path=model_path, control=control)