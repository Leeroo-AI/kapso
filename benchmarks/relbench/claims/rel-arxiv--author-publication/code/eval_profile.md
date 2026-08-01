# Evaluation profile

The registered evaluation suite under `kapso_evaluation/` is immutable, so this measurement profile is kept at repository root rather than modifying the protected directory.

## Mechanics

- Full and fast fidelities both score the complete 39,015-row official validation split; `fraction` and `seed` are manifest metadata only.
- The official RelBench evaluator returns R², MAE, and RMSE. The search score is unweighted row-wise validation R².
- Validation predictions must come from a fit that excludes validation supervision. A separate test fit may use validation labels.
- Output arrays are original-scale finite floats aligned to untouched task order with shapes `(39015,)` and `(39655,)`.

## Input distribution

- Train: 210,769 rows across eight origins from 2018-01-06 through 2021-07-03; 101,886 unique authors.
- Validation: 39,015 unique authors at 2022-01-01. Test: 39,655 unique authors at 2023-01-01.
- Train target mean/std/max are 1.5353/1.2123/67; validation mean/std/max are 1.6806/1.3375/58. Train 90th/99th percentiles are 3/6; validation 90th/99th percentiles are 4/6.
- Prior-paper cold-start rates are 28.14% at validation and 28.75% at test.
- Paper roster median/95th/99th/max sizes are 2/8/19/2,825. Historical-paper cohort-intersection maxima are 2,663 at validation and 1,251 at test.
- Database event dates span 2018-01-01 through 2023-01-01. Citations have 5,295 missing referenced-paper IDs; other measured table columns have no nulls.

## Coverage and discrepancy

Measured axes are origin/regime, target tail, cold start, publication recency and trend, fractional output, team scale, seed-cohort roster overlap, coauthor productivity, citations, categories, text/DOI metadata, and author identity metadata. The supplied coverage claim of roughly 28.8% test cold start is confirmed at 28.75%; the maximum roster size of 2,825 is also confirmed. The expected improvement from cohort-roster/coauthor productivity remains an internal-forward-fold hypothesis and is not assumed from official validation feedback.

## Strata to report

Internal forward-fold diagnostics report count, R², MAE, and RMSE by held origin, cold-start/history status, prior-publication band, and team-size band. Official output exposes only aggregate validation metrics.

## Measured build and internal results

- Uncached all-table feature throughput reached roughly 1.11–1.40 million seed rows/minute; cached matrices load above 2.3 million rows/second. A complete selected-model build from cache finishes in roughly four minutes.
- Four forward folds alternate 183-day strict label embargoes with 365-day shift gaps. Selected calibrated depth-8 CatBoost fold R² values are 0.3938, 0.3752, 0.4019, and 0.4533 versus raw-L2 values 0.3538, 0.3445, 0.3419, and 0.4173.
- Selected fold strata: cold/history counts 56,528/86,039 with R² −0.0056/0.3911; prior-publication bands 0, 1–2, 3–5, and 6+ have R² −0.0056, 0.1286, 0.2522, and 0.3774; maximum-team bands 0–2, 3–10, 11–50, and 51+ have R² 0.0428, 0.2025, 0.3899, and 0.2274.
- Full metadata/category widening lost 0.0081 mean raw-L2 R² and CatBoost full-feature mean R² was 0.3900 versus 0.4067 for core. The one-standard-error rule therefore retained the core block.

## Registered result

Full fidelity scored all 39,015 validation rows and archived `run_0003`: R² 0.6105179324225516, MAE 0.5316690262630472, and RMSE 0.834707113358077. The candidate phase completed in 61.53 seconds from the populated shared feature/selection cache.

After that registered checkpoint, official feedback was not used for selection. A publication-dynamics interaction block improved every internal CatBoost fold by 0.0026–0.0347 R²; its newly selected calibrated fold R² values are 0.3942, 0.3815, 0.4412, and 0.4909. The selected max-team-size 51+ stratum improved internally from 0.2274 to 0.3507.

The interaction checkpoint archived `run_0005` at R² 0.6057523606834528, MAE 0.5296114016104356, and RMSE 0.8397981854606491. A subsequent collaboration-productivity interaction block passed all four internal folds; after correcting the complementarity gate, selected calibrated fold R² is 0.3955/0.3905/0.4553/0.5108 and max-team-size 51+ internal R² is 0.4033.

The structural checkpoint archived `run_0008` at R² 0.6183976918332028, MAE 0.49744138266286914, and RMSE 0.8262203311229112. A later cold/history segmented OOF calibration was rejected (gain 0.00019, SE 0.00032), leaving the selected global affine calibration unchanged.
