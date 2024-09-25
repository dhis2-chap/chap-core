import pandas as pd
import typer
from chap_core.assessment.dataset_splitting import split_test_train_on_period, get_split_points_for_data_set
from chap_core.assessment.multi_location_evaluator import MultiLocationEvaluator
from chap_core.datatypes import ClimateHealthTimeSeries
# from chap_core.external.external_model import ExternalModel
from chap_core.predictor.naive_predictor import MultiRegionNaivePredictor
from chap_core.reports import HTMLReport
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from tests.test_external_model_evaluation_acceptance import ExternalModelMock
from chap_core._legacy_dataset import IsSpatioTemporalDataSet


def load_data_set(data_set_filename: str) -> IsSpatioTemporalDataSet:
    assert data_set_filename.endswith('.csv')
    return DataSet.from_pandas(pd.read_csv(data_set_filename), ClimateHealthTimeSeries)


def main(
        python_script_filename: str,
        dataset_name: str,
        data_set_filename: str,
        output_filename: str
):
    external_model = ExternalModelMock(python_script_filename, adaptors=None)
    data_set = load_data_set(data_set_filename)
    evaluator = MultiLocationEvaluator(model_names=['external_model', 'naive_model'], truth=data_set)
    split_points = get_split_points_for_data_set(data_set, max_splits=5)

    for (train_data, future_climate_data, future_truth) in split_test_train_on_period(data_set, split_points,
                                                                                      future_length=None,
                                                                                      include_future_weather=True):
        predictions = external_model.get_predictions(train_data, future_climate_data)
        evaluator.add_predictions('external_model', predictions)
        naive_predictor = MultiRegionNaivePredictor()
        naive_predictor.train(train_data)
        naive_predictions = naive_predictor.predict(future_climate_data)
        evaluator.add_predictions('naive_model', naive_predictions)

    results = evaluator.get_results()
    report = HTMLReport.from_results(results)
    report.save(output_filename)


if __name__ == "__main__":
    typer.run(main)
    # test CLI from project root:
    # python3 scripts/ext_model_evaluation.py tests/mock_predictor_script.py my_cool_data example_data/hydro_met_subset.csv tests/tmp_report.html
