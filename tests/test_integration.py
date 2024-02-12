from climate_health.main import assess_model_on_csv_data, PlaceholderModel


def test_full_run():
    data_file = "../example_data/data.csv"
    report = assess_model_on_csv_data(data_file, 0.5, PlaceholderModel())
    print(report.text)

