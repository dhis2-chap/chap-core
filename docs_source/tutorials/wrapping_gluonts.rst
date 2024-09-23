Wrapping GluonTS models
-----------------------
GluonTS provides a set of models that can be used for probabilistic time-series forecasting.
Here, we show how we can wrap these models into CHAP models, to enable using them on spatio-temporal data and to evalutate them against other models.

We will use the `DeepAREstimator` model from GluonTS, which is a deep learning model based on an RNN architecture. For this simple example we use
a model that does not take weather into account, but only the the auto-regressive time series data.
Let's start by loading the data and the model.

.. code-block:: python

    from climate_health.data.datasets import ISIMIP_dengue_harmonized
    from gluonts.torch import DeepAREstimator
    from gluonts.torch.distributions import NegativeBinomialOutput

    # Load the data
    data = ISIMIP_dengue_harmonized['vietnam']

    # Define the DeepAR model
    n_locations = len(data.locations)
    prediction_length = 4
    deep_ar =  DeepAREstimator(
        num_layers=2,
        hidden_size=24,
        dropout_rate=0.3,
        num_feat_static_cat=1,
        scaling=False,
        embedding_dimension=[2],
        cardinality=[n_locations],
        prediction_length=prediction_length,
        distr_output=NegativeBinomialOutput(),
        freq='M')

     # Wrap the model in a CHAP model

     from climate_health.adapters.gluonts import GluonTSEstimator

     model = GluonTSEstimator(gluonts_model, data)


The model now is a chap compatible model and we can run our evaluation pipeline on it.

.. code-block:: python

    from climate_health.evaluation import evaluate_model

    evaluate_model(model, data, prediction_length=4, n_test_sets=8, report_filename='gluonts_deepar_results.csv')



