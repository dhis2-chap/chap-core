"""
Script for full run of training a model and assessing it on a dataset.
"""
from chap_core.main import assess_model_on_csv_data, PlaceholderModel


if __name__ == "__main__":
    report = assess_model_on_csv_data("../example_data/data.csv", 0.5, PlaceholderModel())
    print(report.text)
