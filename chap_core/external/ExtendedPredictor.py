from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import Samples
import pandas as pd


class ExtendedPredictor(ConfiguredModel):
    def __init__(self, configured_model: ConfiguredModel, desired_scope):
        self._config_model = configured_model
        self._desired_scope = desired_scope

    def train(self, train_data: DataSet, extra_args=None):
        self._config_model.train(train_data, extra_args)
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        historic_df = historic_data.to_pandas()
        future_df = future_data.to_pandas()

        # Drop 'parent' column if present - it's metadata for hierarchical aggregation,
        # not a predictive feature, and can contain non-numeric placeholders like '-'
        if "parent" in historic_df.columns:
            historic_df = historic_df.drop(columns=["parent"])
        if "parent" in future_df.columns:
            future_df = future_df.drop(columns=["parent"])

        model_information = self._config_model.model_information

        assert model_information is not None

        min_pred_length = model_information.min_prediction_length
        max_pred_length = model_information.max_prediction_length

        assert self._desired_scope >= min_pred_length

        remaining_time_periods = self._desired_scope
        predictions = pd.DataFrame()

        future_time_idx = 0
        unique_time_periods = future_df["time_period"].unique()

        while remaining_time_periods > 0:
            steps_to_predict = min(max_pred_length, remaining_time_periods)

            # Slice by time period values to include all locations
            time_periods_to_select = unique_time_periods[future_time_idx : future_time_idx + steps_to_predict]
            future_slice = future_df[future_df["time_period"].isin(time_periods_to_select)]

            new_prediction = self._config_model.predict(
                DataSet.from_pandas(historic_df), DataSet.from_pandas(future_slice)
            )
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

            future_time_idx += newly_predicted
            remaining_time_periods -= newly_predicted

            historic_df = self.update_historic_data(historic_df, new_prediction_pandas, newly_predicted)

        # Remove duplicate time periods, keeping the last (most recent) prediction
        # This is important because later predictions are more accurate (shorter prediction horizon)
        predictions = predictions.drop_duplicates(subset=["time_period", "location"], keep="last")
        predictions = predictions.reset_index(drop=True)

        return DataSet.from_pandas(predictions, Samples)

    def update_historic_data(self, historic_data_pandas, predictions_pandas, num_predictions):
        # Select rows by time period, not by row index, to include all locations
        unique_time_periods = predictions_pandas["time_period"].unique()
        time_periods_to_add = unique_time_periods[:num_predictions]
        new_rows = predictions_pandas[predictions_pandas["time_period"].isin(time_periods_to_add)].copy()

        # Find all columns that start with "sample_"
        sample_cols = [col for col in new_rows.columns if col.startswith("sample_")]

        # Replace them with a single column "disease_cases" as the average
        new_rows["disease_cases"] = new_rows[sample_cols].mean(axis=1)

        # Drop the original sample columns
        new_rows = new_rows.drop(columns=sample_cols)

        # Concatenate with historic data
        updated_history = pd.concat([historic_data_pandas, new_rows], ignore_index=True)
        return updated_history
