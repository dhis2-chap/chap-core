from chap_core.models.configured_model import ConfiguredModel
from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import pandas as pd


class ExtendedPredictor(ConfiguredModel):

    def __init__(self, configured_model : ExternalModel, desired_scope):
        self._config_model = configured_model
        self._desired_scope = desired_scope

    def train(self, train_data: DataSet, extra_args=None):
        self._config_model.train(train_data, extra_args)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        historic_df = historic_data.to_pandas()
        future_df = future_data.to_pandas()

        model_information = self._config_model.model_information

        assert model_information is not None

        min_pred_length = model_information.min_prediction_length
        max_pred_length = model_information.max_prediction_length

        assert self._desired_scope >= min_pred_length

        remaining_time_periods = self._desired_scope
        predictions = pd.DataFrame()

        future_idx = 0

        while remaining_time_periods > 0:
            steps_to_predict = min(max_pred_length, remaining_time_periods)

            future_slice = future_df.iloc[future_idx: future_idx + steps_to_predict]

            new_prediction = self._config_model.predict(DataSet.from_pandas(historic_df),
                                                        DataSet.from_pandas(future_slice))
            new_prediction_pandas = new_prediction.to_pandas()
            predictions = pd.concat([predictions, new_prediction_pandas])
            if remaining_time_periods > max_pred_length:
                if remaining_time_periods <= 2 * max_pred_length:
                    # shift window so last prediction is exactly max_pred_length
                    newly_predicted = remaining_time_periods - max_pred_length
                else:
                    # predict max normally
                    newly_predicted = max_pred_length
            else:
                # last prediction
                newly_predicted = remaining_time_periods

            future_idx += newly_predicted
            remaining_time_periods -= newly_predicted

            historic_df = self.update_historic_data(historic_df, new_prediction_pandas,
                                                             newly_predicted)

        # Remove duplicate time periods, keeping the last (most recent) prediction
        # This is important because later predictions are more accurate (shorter prediction horizon)
        predictions = predictions.drop_duplicates(subset=['time_period', 'location'], keep='last')
        predictions = predictions.reset_index(drop=True)

        return DataSet.from_pandas(predictions)

    def update_historic_data(self, historic_data_pandas, predictions_pandas, num_predictions):
        new_rows = predictions_pandas.head(num_predictions).copy()

        # Find all columns that start with "sample_"
        sample_cols = [col for col in new_rows.columns if col.startswith("sample_")]

        # Replace them with a single column "disease_cases" as the average
        new_rows["disease_cases"] = new_rows[sample_cols].mean(axis=1)

        # Drop the original sample columns
        new_rows = new_rows.drop(columns=sample_cols)

        # Concatenate with historic data
        updated_history = pd.concat([historic_data_pandas, new_rows], ignore_index=True)
        return updated_history


