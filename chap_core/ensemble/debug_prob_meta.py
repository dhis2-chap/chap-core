from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from chap_core.api_types import RunConfig
from chap_core.assessment.metrics.crps import CRPSMetric
from chap_core.cli_endpoints._common import (
    discover_geojson,
    load_dataset_from_csv,
)
from chap_core.datatypes import FullData, Samples
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass
class BaseModelInfo:
    """Holder info om en basemodell vi vil inspisere."""

    name: str  # bare et lesbart navn
    path_or_url: str  # lokal sti eller GitHub-URL


def split_inner_train_val(
    data: DataSet,
    inner_val_periods: int = 12,
) -> tuple[DataSet, DataSet]:
    """
    Kopi av logikken i EnsembleEstimator._split_inner_train_val,
    men som ren funksjon.
    """
    df = data.to_pandas()
    all_periods = df["time_period"].dropna().astype(str).sort_values().unique()

    if len(all_periods) <= inner_val_periods:
        split_idx = len(all_periods) // 2
    else:
        split_idx = len(all_periods) - inner_val_periods

    train_periods = set(all_periods[:split_idx])
    val_periods = set(all_periods[split_idx:])

    df_train = df[df["time_period"].astype(str).isin(train_periods)].copy()
    df_val = df[df["time_period"].astype(str).isin(val_periods)].copy()

    inner_train = DataSet.from_pandas(df_train, FullData, fill_missing=True)
    val_data = DataSet.from_pandas(df_val, FullData, fill_missing=True)
    return inner_train, val_data


def _load_estimator_from_path_or_url(path_or_url: str):
    """
    Laster en estimator-klasse gitt en sti eller GitHub-URL,
    på samme måte som i CLI, med samme defaults som RunConfig().
    """
    run_config = RunConfig()  # samme defaults som brukes i CLI

    tpl: ModelTemplate = ModelTemplate.from_directory_or_github_url(
        path_or_url,
        base_working_dir=CHAP_RUNS_DIR,
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
    )
    EstimatorCls = tpl.get_model(None)
    est = EstimatorCls()  # type: ignore[call-arg]
    return est


def inspect_base_model_samples_on_val(
    dataset_csv: str,
    geojson_path: str | None,
    base_models: Sequence[BaseModelInfo],
    inner_val_periods: int = 12,
    max_locations: int = 3,
    max_rows_per_loc: int = 20,
) -> None:
    """
    Laster dataset fra CSV, splitter i (inner_train, val),
    trener hver basemodell på inner_train, og skriver ut
    Samples.to_pandas() for noen locations på valideringssettet.

    Bruk dette for å sjekke om basemodellene faktisk gir flere samples
    (sample_*-kolonner) på val-delen.
    """
    if geojson_path is None:
        geojson_path = discover_geojson(dataset_csv)

    print(f"[DEBUG] Laster dataset fra {dataset_csv} med geojson={geojson_path}")
    ds = load_dataset_from_csv(dataset_csv, geojson_path, column_mapping=None)

    print("[DEBUG] Splitter dataset i inner_train / val")
    inner_train, val_data = split_inner_train_val(ds, inner_val_periods=inner_val_periods)

    locations = list(val_data.locations())
    if not locations:
        print("[DEBUG] Ingen locations i val_data – stopp")
        return

    locations = locations[:max_locations]

    for bm in base_models:
        print("=" * 80)
        print(f"[DEBUG] Basemodell: {bm.name} ({bm.path_or_url})")
        est = _load_estimator_from_path_or_url(bm.path_or_url)

        print("[DEBUG] Trener på inner_train...")
        pred = est.train(inner_train).predict(inner_train, val_data)  # DataSet[Samples]

        for loc in locations:
            samples_loc: Samples = pred[loc]
            df_pred = samples_loc.to_pandas()

            print("-" * 60)
            print(f"[DEBUG] Location: {loc}")
            print("[DEBUG] Kolonner:", list(df_pred.columns))
            if any(c.startswith("sample_") for c in df_pred.columns):
                sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
                print(f"[DEBUG] Antall sample_-kolonner: {len(sample_cols)}")
            else:
                print("[DEBUG] Ingen 'sample_*'-kolonner funnet")

            print(df_pred.head(max_rows_per_loc))


def _samples_matrix_from_df(df_pred: pd.DataFrame) -> np.ndarray:
    """
    Tar en df fra Samples.to_pandas() og returnerer en (T, S)-matrise
    der S er antall sample_* kolonner.
    """
    sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
    if not sample_cols:
        raise ValueError("Fant ingen 'sample_*' kolonner i prediksjons-DataFrame")

    # Sorter kolonnene etter nummeret i sample_i for deterministisk rekkefølge
    sample_cols_sorted = sorted(
        sample_cols,
        key=lambda c: int(c.split("_")[1]),  # "sample_123" -> 123
    )
    arr = df_pred[sample_cols_sorted].to_numpy()  # shape (T, S)
    return arr


def build_ensemble_from_base_samples(
    base_preds: Sequence[DataSet[Samples]],
    weights: np.ndarray,
    val_data: DataSet,
) -> DataSet[Samples]:
    """
    Bygger et ensemble av samples ut fra basemodellenes Samples på val_data.

    - Alle basemodeller kan ha forskjellig antall samples S_m.
    - Vi trunkerer til S_min = min_m S_m for å få felles S.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("weights må være >= 0")
    s = weights.sum()
    if s <= 0:
        raise ValueError("sum(weights) må være > 0")
    weights = weights / s

    result: dict[Any, Samples] = {}

    for loc in val_data.locations():
        tp = val_data[loc].time_period

        mats = []
        for ds in base_preds:
            samp: Samples = ds[loc]
            df_pred = samp.to_pandas()
            arr = _samples_matrix_from_df(df_pred)  # (T, S_m)
            mats.append(arr)

        # Finn felles antall samples S_min
        S_list = [a.shape[1] for a in mats]
        S_min = min(S_list)
        if S_min == 0:
            raise ValueError("Ingen samples i en eller flere modeller")

        # Trunker alle til (T, S_min)
        mats_trunc = [a[:, :S_min] for a in mats]

        # Stack til (M, T, S_min)
        mats_arr = np.stack(mats_trunc, axis=0)  # (M, T, S)

        # Vekt langs M-aksen
        ensemble_mat = np.tensordot(weights, mats_arr, axes=([0], [0]))  # (T, S_min)

        result[loc] = Samples(samples=ensemble_mat, time_period=tp)

    return DataSet(result)


def _wide_samples_to_long_fc_df(fc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Konverterer DataFrame med sample_*-kolonner til long-format
    med kolonnene: location, time_period, horizon_distance, sample, forecast.
    """
    sample_cols = [c for c in fc_df.columns if c.startswith("sample_")]
    id_cols = [c for c in fc_df.columns if c not in sample_cols]

    if not sample_cols:
        raise ValueError("Forventer sample_*-kolonner i fc_df")

    long_df = fc_df.melt(
        id_vars=id_cols,
        value_vars=sample_cols,
        var_name="sample",
        value_name="forecast",
    )
    # sample-kolonnen er f.eks. 'sample_3' -> gjør om til int 3
    long_df["sample"] = long_df["sample"].astype(str).str.split("_").str[1].astype(int)
    return long_df


def compute_crps_for_candidate_weights(
    dataset_csv: str,
    geojson_path: str | None,
    base_models: Sequence[BaseModelInfo],
    weights: np.ndarray,
    inner_val_periods: int = 12,
) -> pd.DataFrame:
    """
    Tren hver basemodell på inner_train, hent Samples på val,
    bygg et ensemble med gitt weights, og beregn global CRPS
    på val-delen ved hjelp av CHAP sin CRPSMetric.
    """
    if geojson_path is None:
        geojson_path = discover_geojson(dataset_csv)

    ds = load_dataset_from_csv(dataset_csv, geojson_path, column_mapping=None)
    inner_train, val_data = split_inner_train_val(ds, inner_val_periods=inner_val_periods)

    base_preds: list[DataSet[Samples]] = []
    for bm in base_models:
        est = _load_estimator_from_path_or_url(bm.path_or_url)
        pred = est.train(inner_train).predict(inner_train, val_data)
        base_preds.append(pred)

    ensemble_ds = build_ensemble_from_base_samples(
        base_preds=base_preds,
        weights=weights,
        val_data=val_data,
    )

    metric = CRPSMetric()

    obs_df = val_data.to_pandas()
    fc_df = ensemble_ds.to_pandas()

    # CRPSMetric forventer bl.a. 'horizon_distance'. Ensemblet vårt er 0-step ahead
    # på val-delen, så vi setter horizon_distance = 0 hvis den mangler.
    if "horizon_distance" not in fc_df.columns:
        fc_df["horizon_distance"] = 0

    # Konverter wide samples (sample_*) til long-format med 'sample' og 'forecast'
    fc_long = _wide_samples_to_long_fc_df(fc_df)

    df_crps = metric.get_global_metric(obs_df, fc_long)
    return df_crps
