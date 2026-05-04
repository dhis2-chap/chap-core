from __future__ import annotations

import numpy as np

from chap_core.ensemble.debug_prob_meta import (
    BaseModelInfo,
    compute_crps_for_candidate_weights,
    inspect_base_model_samples_on_val,
)


def main():
    dataset_csv = "ensemble/data/chap_LAO_admin1_monthly.csv"
    geojson_path = None
    inner_val_periods = 12

    base_models = [
        BaseModelInfo(
            name="rwanda_sarimax",
            path_or_url="https://github.com/chap-models/rwanda_sarimax",
        ),
        BaseModelInfo(
            name="chap_auto_ewars",
            path_or_url="https://github.com/chap-models/chap_auto_ewars",
        ),
        BaseModelInfo(
            name="INLA_baseline_model",
            path_or_url="https://github.com/chap-models/INLA_baseline_model",
        ),
    ]

    print("\n=== 1) INSPEKTERER SAMPLES FRA BASEMODELLER PÅ VAL ===\n")
    inspect_base_model_samples_on_val(
        dataset_csv=dataset_csv,
        geojson_path=geojson_path,
        base_models=base_models,
        inner_val_periods=inner_val_periods,
        max_locations=2,
        max_rows_per_loc=10,
    )

    print("\n=== 2) TESTER CRPS FOR LIK VEKT-ENSEMBLE PÅ VAL ===\n")
    weights = np.array([1.0, 1.0, 1.0], dtype=float)
    df_crps = compute_crps_for_candidate_weights(
        dataset_csv=dataset_csv,
        geojson_path=geojson_path,
        base_models=base_models,
        weights=weights,
        inner_val_periods=inner_val_periods,
    )
    print(df_crps)


if __name__ == "__main__":
    main()
