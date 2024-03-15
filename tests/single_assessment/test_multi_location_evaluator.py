import pytest

from climate_health.assessment.multi_location_evaluator import MultiLocationEvaluator
from ..data_fixtures import full_data, good_predictions, bad_predictions


@pytest.mark.xfail
def test_multi_location_evaluator(full_data, good_predictions, bad_predictions):
    evaluator = MultiLocationEvaluator(model_names=['bad_model', 'good_model'],
                                       truth=full_data)
    evaluator.add_predictions('good_model', good_predictions)
    evaluator.add_predictions('bad_model', bad_predictions)
    results = evaluator.get_results()

    for i in range(len(results['good_model']['mae'])):
        assert results['good_model']['mae'][i] < results['bad_model']['mae'][i]
