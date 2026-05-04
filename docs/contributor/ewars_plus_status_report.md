---
title: ewars_Plus integration status report
date: 2026-05-04
---

*Investigation period: 2026-04-23 → 2026-04-30. Stack: `chap-core` master + R model `maquins/ewars_plus_api:Upload` overlaid with the CHAP patch. Test payloads: 1-district Malawi (`~/Downloads/EWARS.json`, 114 weekly periods, single org-unit `A2Kgu7zMgJr` "Phalombe-DHO") and a synthetic 5-district variant (`EWARS_5copies_v3.json`).*

---

## 1. Executive summary

The `ewars_plus` model now reaches a chap-core `BackTest` artefact for **about 80% of the backtest windows it is asked to compute** on the 1-district Malawi payload. Before this work, *zero* windows completed end-to-end — the very first call failed with an opaque `AttributeError`. Five distinct defects were diagnosed, reproduced with unit tests, and fixed across two repositories. One defect remains as a workaround rather than a true fix (predictions are carry-forward-imputed because the model produces sparse output), and one defect is unfixable from outside the R model itself: the R server stalls non-deterministically inside INLA on roughly one window in every 10 of a sequential backtest. With the merged `curl --max-time` change, that stall now fails fast (~30 min) with a diagnosable error rather than blocking the worker forever.

**Net assessment.** The code path is now operationally usable for end-to-end smoke tests on any payload, but it is **not** ready to produce a meaningful evaluation on 1-district inputs (carry-forward bias) and is **not** ready for unattended long backtests until the INLA stall is addressed upstream. Multi-district inputs were not end-to-end-validated in this session (every attempt was time-bounded or interrupted).

---

## 2. Issues found and resolved

### 2.1 Wrapper layer (`dhis2-chap/ewars_plus_python_wrapper`)

| ID | Symptom | Root cause | Fix | PR |
|---|---|---|---|---|
| **B1** | `AttributeError: 'str' object has no attribute 'get'` from `change_prediction_format_to_chap` whenever the R server returned a non-list error body. | The function iterated `json_data` blindly — a dict yields its string keys, a bare-string body yields characters. | Added `_check_api_response()` that raises `RuntimeError: EWARS API error from {endpoint}: {body}` when the body is `{error: ...}`; added an `isinstance(json_data, list)` guard before iterating. | [#2](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/2) merged (CLIM-614) |
| **B4** | Worker hung indefinitely (1h+ at 0% CPU) when the R server stalled mid-CV. | All three curl invocations omitted `--max-time`; the wrapper's `subprocess.run(...)` blocked on the unresponsive curl. | `curl --max-time {1800,900,120}` for `/Ewars_run`, `/Ewars_predict`, `/retrieve_predicted_cases` respectively. On timeout `curl` exits 28 and the wrapper raises a clean `RuntimeError`. | [#3](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/3) merged (CLIM-618) |
| **B5a** | chap-core rejected predictions with `Periods must be consecutive`. | The R model only fills `predicted_cases` for a sparse subset of the requested forecast weeks (e.g. it returned `(W20, W24, W26)` when `future_data` asked for `(W19, W20, W21)`). The wrapper used `df.groupby('location').head(n_to_predict)` which took whichever populated rows happened to come first, regardless of which weeks they covered. | Added `requested_periods=[(year, week), ...]` argument to `change_prediction_format_to_chap`, populated from the `future_data` CSV. Output is restricted to that intersection, sorted ascending. | [#5](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/5) merged |
| **B5b** | After PR #5, wrapper crashed with `pd.concat(...): No objects to concatenate`. | `predict_wrapper` makes two `/Ewars_predict` calls per backtest window. The first is an offset-discovery probe that needs to see whichever weeks the model populated; PR #5 made it filter to `requested_periods`, which collapsed to empty when the model and the request didn't overlap. | Added `apply_period_filter=True` argument to `predict()`, pass `False` for the offset-discovery call. | [#6](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/6) merged |
| **B5c** | After PR #6, chap-core rejected predictions with `PeriodRange(2024W19..2024W21) != PeriodRange(2024W20..2024W20)`. | chap-core's `Evaluation.from_samples_with_truth` merges prediction and truth by exact `PeriodRange` equality. A 1-week prediction for a 3-week truth window does not satisfy the merge. | Added `align_to_future_periods()` that pads the prediction frame to one row per `(location, year, week)` tuple in `future_data`. | [#7](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/7) merged |
| **B5d** | After PR #7's NaN padding, chap-core rejected predictions with `Samples are not finite`. | `Samples.from_pandas` enforces `np.isfinite(samples).all()`. | Switched padding from NaN to `groupby('location').ffill().bfill()` so missing weeks inherit the nearest populated week's samples. Raises a clear `RuntimeError` if any location has zero populated rows. | [#8](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/8) merged |
| **Tests** | None of the helpers had unit coverage. | — | 12-test pytest module covering `change_prediction_format_to_chap` shape validation, `run_command` exit codes (including `28` for timeout), and `_check_api_response` JSON-error handling. | [#4](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/4) **open** |

### 2.2 R model layer (chap-core PR [#315](https://github.com/dhis2-chap/chap-core/pull/315), open)

The R model code is shipped as an unpublished Docker image (`maquins/ewars_plus_api:Upload`). Patches are applied as a one-layer overlay defined in `external_models/ewars_plus_api_patch/`.

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| **B2** | `<simpleError in get_preds(aa): names do not match previous names>` from `Lag_Model_selection_ewars_By_District_api.R` after CV completed. | Lag selection picks a different optimal lag per year (e.g. 2022 → `_LAG12`, 2023 → `_LAG10`, 2024 → `_LAG7`). The 39 per-fold RDS files therefore had differently-named columns; base R `rbind` via `foreach(.combine = rbind)` refused to stack them. | Replace with `dplyr::bind_rows(lapply(seq_len(nrow(all_files_Cv)), get_preds))`. Missing columns become NA; downstream code does not read the lag-suffixed columns so the extra NAs are harmless. |
| **B3** | `<simpleError ... non-conformable arrays>` from `DBII_predictions_Vectorized_API.R` during `/Ewars_predict`. | The idiom `idx.end_all <- foreach(aa = Prospective_Data_with_inla_grp$week, .combine = c) %do% which(endemic_channel_Use$week == aa)` silently drops weeks not present in the endemic table. The threshold matrix shrank, then the elementwise comparison with the predicted-rate matrix failed. | Replace with `idx.end_all <- match(Prospective_Data_with_inla_grp$week, endemic_channel_Use$week)`. Length is preserved; unmatched weeks become NA; comparison harmlessly produces NA in those rows. |
| **Tests** | None for the R patches. | — | Two synthetic-fixture R unit tests (`test_bind_rows_assembly.R`, `test_idx_end_match.R`) plus a docker-build smoke (`tests/test_ewars_plus_api_patch.sh`). |
| **Docs** | The full call flow had never been written down. | — | `docs/contributor/ewars_plus_backtest_flow.md` — sequence diagram + bug-site legend. |

---

## 3. Issues that remain

### 3.1 Non-deterministic INLA stalls (highest priority)

**Symptom.** During CV inside `/Ewars_run`, the R process drops to 0% CPU and prints nothing further. After 30 min the wrapper's `curl --max-time 1800` cancels the request and the Celery task fails with `curl: (28) Operation timed out`.

**Observed behaviour over three sequential 10-window backtests of the same Malawi payload:**

| Run | Windows completed | Stall window |
|---|---|---|
| First debug-loop run | 5 | window 6 (`historic_data_2024-08-26.csv`) |
| Second debug-loop run | 8 | window 9 (`historic_data_2024-11-18.csv`) |
| Initial v3 5-district run (interrupted by user) | 0 (window 1, district 2 reached when stopped) | n/a |

The stall position changes between runs — strong evidence this is not data-specific but an INLA-internal convergence/concurrency phenomenon. With CLIM-618 in place the worker is no longer wedged, but each stalled window still costs ~30 min of compute and produces no predictions for that fold.

**What is *not* known.** Whether this reproduces on a native amd64 host (current observations are all on Mac qemu emulation), whether `inla.set.control.compute(safe = TRUE)` or other INLA tuning would help, and whether the stalls correlate with any data feature of the affected slice.

**Recommended next step.** Reproduce on a native amd64 chap-core host and, if present, instrument R-side: trace which call inside the CV loop fails to return (likely `inla.posterior.sample`, `inla.posterior.sample.eval`, or one of the `apply()` calls building `preds`). Possibly tighten `num.threads` from `"1:1"` to `"1:1"` explicitly or add `control.inla = list(strategy = "laplace")`. None of this is fixable from chap-core or the wrapper.

### 3.2 Sparse `predicted_cases` output (model-design, not bug)

**Symptom.** The R model populates `predicted_cases` for an irregular subset of the requested forecast weeks (lag warm-up + horizon decisions). For one observed window, the model returned `predicted_cases` for `(W20, W24, W26, W27, W28)` while `future_data` requested `(W19, W20, W21)`. Without a wrapper-side filter chap-core sees the wrong weeks; with it, several requested weeks have no model prediction at all.

**Current workaround (PR #8).** `align_to_future_periods` carries forward (`ffill().bfill()`) the nearest populated week's samples into the missing rows. This satisfies chap-core's three-way constraint (consecutive periods, exact-equality merge, finite samples) **but biases the evaluation**: multiple weeks share a single point estimate. Window-level metrics on such windows are not statistically meaningful.

**Why this matters more on 1-district inputs.** With one district the model often produces only 1–3 populated weeks per request, so most cells are imputed. With multi-district inputs the spatial-pooling step usually produces fuller forecast horizons; that's not been validated end-to-end this session.

**Recommended next step.** Investigate the R-side `Prediction distance` config and `predict_wrapper`'s offset arithmetic. If the offset is computed correctly we should be asking the R server for the precise weeks we need; the fact that we get `predicted_cases` for `(W20, W24, W26, W27, W28)` instead of contiguous `(W19, W20, W21)` suggests either a config mismatch or a bug in the R model's prediction-window selection.

### 3.3 Stateful R container forces re-training per window

**Symptom.** chap-core's external-model contract calls `predict` once per backtest window. The wrapper's `predict` entry point unconditionally calls `train()` first because the R server stores trained state on its own filesystem and cannot pickle it back to the wrapper.

**Cost.** ~6–15 min CV per window per district. A 10-window 1-district backtest is ~60–150 min wall-time on emulation; multi-district scales linearly.

**Recommended next step.** Either (a) make the R server stateless (carry the trained state in HTTP responses), (b) introduce named training sessions (`/Ewars_run?session=window-7`), or (c) have the R server return a tarball of the RDS files and the wrapper push it back on each predict. None of these are wrapper-only fixes.

### 3.4 1-district payloads are not a fair evaluation surface

The model was designed for multi-district spatial pooling. With n=1 the spatial component is degenerate and `predicted_cases` is sparse. The 1-district Malawi payload has been useful as a smoke test for the pipeline but should not be used to assess model quality. The synthetic 5-district variant (`EWARS_5copies_v3.json`) reached `done district:2 of 5` in window 1 before being stopped; a full 5-district run is the next obvious validation experiment but takes ≥5 hours of wall-time on emulation.

### 3.5 Upstream code is unpublished

The `Lag_Model_selection_ewars_By_District_api.R` and `DBII_predictions_Vectorized_API.R` files baked into `maquins/ewars_plus_api:Upload` are not in any public GitHub repository found this session. The closest relatives are `maquins/ewars_Plus` (Shiny app) and `maquins/EWARS_Plus_Server` (different code). The chap-core patch directory (`external_models/ewars_plus_api_patch/`) is therefore a permanent fork rather than a temporary patch awaiting upstream merge. Any future image bump from the upstream maintainer needs re-application of the two R patches and re-validation against the failure payloads documented in CLIM-615 / CLIM-617.

---

## 4. Where the deliverables live

| Item | Location | State |
|---|---|---|
| chap-core R-patch + tests + docs | `dhis2-chap/chap-core` PR [#315](https://github.com/dhis2-chap/chap-core/pull/315) | open, mergeable |
| Wrapper response-handling fixes | `ewars_plus_python_wrapper` PRs #2, #3, #5, #6, #7, #8 | merged into `modeling_app_test` |
| Wrapper unit tests | `ewars_plus_python_wrapper` PR [#4](https://github.com/dhis2-chap/ewars_plus_python_wrapper/pull/4) | open |
| Backtest call-flow doc | `chap-core/docs/contributor/ewars_plus_backtest_flow.md` (in PR #315) | committed |
| Jira | CLIM-614 Done, CLIM-615 Done, CLIM-617 Selected for Development, CLIM-618 Selected for Development | tracked |

---

## 5. Recommendations, in priority order

1. **Investigate the INLA stall on native amd64 hardware** before assuming it's a Mac qemu artefact. If it reproduces, instrument R-side and consider INLA tuning. This is the single biggest blocker to any unattended deployment.
2. **Run the 5-district payload end-to-end** on a native host. Multi-district is what the model is built for; a clean 5-district success would let us retire the carry-forward workaround for that input shape.
3. **Decide a posture on 1-district inputs**: either document them as smoke-test-only (recommended) or implement an explicit downstream warning when chap-core sees imputed predictions (the wrapper could emit a marker column the evaluator surfaces).
4. **Land PR #315 and PR #4.** The R patches are validated; the wrapper tests guard the helpers. Both are low-risk merges and unblock the deployment story.
5. **Address the train-per-window cost** (Issue 3.3) only after 1 and 2 are answered. Without those, a fast pipeline still produces unreliable results.
