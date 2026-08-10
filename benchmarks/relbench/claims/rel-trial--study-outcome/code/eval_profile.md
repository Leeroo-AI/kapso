# Evaluation profile

## Measurement mechanics

The immutable grader invokes `python main.py` for full fidelity and `python main.py --debug` for fast fidelity. Both modes score all 960 validation rows; `--fraction` and `--seed` are manifest metadata and do not change the scored rows. The official task evaluator computes average precision, accuracy, F1, and ROC-AUC, and the configured score of record is validation ROC-AUC. Full runs are archived with both positional prediction vectors. The grader controls only the run mode, output directory, timeout, sanitized cache, and task identity; candidate feature construction, inference, calibration, rounds, and blending remain candidate-controlled.

## Input distribution

- Train: 11,994 unique trials, 19 annual cohorts from 2001-01-05 through 2019-01-01, positive rate 0.63757. Cohort sizes range from 2 to 1,153; 2016--2019 contain 1,093, 1,153, 1,128, and 1,093 rows.
- Validation: 960 unique trials at 2020-01-01, 561 positives, positive rate 0.58438.
- Test: 825 unique trials at 2021-01-01. Labels are absent.
- The eligible-primary historical result corpus contains exactly 14,164 trials and 14,164 report-date episodes. Every trial has one eligible report date. Eligible analysis count has median 1, 90th percentile 5, 95th percentile 8, and maximum 703.
- Exact reconstruction matches all 11,994 train labels and all 960 validation labels. Among train rows, positive trials average 3.44396 eligible analyses and negative trials average 1.85185.
- All 13,779 seed rows have zero legal pre-seed rows in outcomes, outcome analyses, drop/withdrawals, and reported event totals.
- Registration coverage is broad: eligibility 13,779/13,779; design 13,774/13,779; sponsors 13,779/13,779; conditions 12,492/13,779; interventions 8,381/13,779; facilities 12,732/13,779.
- Validation differs from train in label rate and is a single forward cohort. Test is another single cohort, so design selection must use recent forward train folds rather than the official validation value.

## Coverage axes

The feature and slice axes are seed year, result-history window, exact-entity support and cold start, site-count richness, country breadth, sponsor class, source/source class, phase and purpose, intervention/condition availability, enrollment scale, trial age, eligibility-text complexity, semantic-neighbor support, and predicted analysis multiplicity. Small timestamp cohorts below 20 require neutral within-cohort transforms. The solution's measured coverage claims were confirmed. Its assumed residual value of semantic neighbors and predicted multiplicity remains gated by the 2016--2019 forward-fold comparison.

## Critical path

The score-bounding artifact for this iteration is the annual timestamp-censored entity-semantic diffusion matrix. The inherited BioClinical benchmark reached 178.6 short documents/s, so the target for approximately 60,000 condition, intervention, and sponsor names is at least 100 entities/s; the propagation target is at least 500 seed routes/s. Consumers are deliberately second: forward gating and two-lineage residual fitting begin only after the diffusion cache and precision diagnostics are durable.

## Iteration 2 coverage audit

The immutable measurement mechanics and split profile above remain unchanged. The new coverage axes are entity type, exact versus bridged support, all-history versus five-year half-life evidence, diffusion depth, bridge threshold, route availability, script family, facility country, and sparse versus rich sites. Test labels remain absent and no validation score chooses these rules.

The full profile resolved the bridge-precision, multilingual-name, and facility-locality assumptions. Non-ASCII name rates are 0% for condition/intervention, 7.17% for sponsors, and 8.73% for facilities, while non-Latin letter rates are at most 0.029%. A 64-pair training-only audit found mean accepted cosine 0.830/0.840/0.870/0.959 by route, at least 93.8% script agreement, 100% sponsor-agency agreement, and 85.9% facility locality agreement. Entity encoding ran at 2,807--3,563 names/s and the complete 13,779 by 648 diffusion matrix took 117.54 seconds, exceeding both planned throughput targets.

The diffusion tree block did not pass the endpoint blend gate: pure mart mean forward AUC was 0.705440, while the strongest diffusion mean was 0.704638, and every diffusion blend lost to weight 1.0 mart. A separate matching 2017--2019 OOF structural-diversity check selected frozen graph weight 0.40 after improving all three folds from 0.695044/0.717566/0.713380 to 0.697557/0.722496/0.716810. That rule was fixed before another official validation measurement.

## Required reporting

Internal results are reported per forward cohort and by history/site-richness strata where class support permits. Official validation is diagnostic only and cannot alter features, rounds, model choice, or blend.

## Final measurement

The frozen run used combined and title-section semantic neighbors, 263 LightGBM trees, logistic C=0.02, and LightGBM blend weight 0.85. Its 2016--2019 blended fold AUCs were 0.696398, 0.696961, 0.718819, and 0.714537; pooled AUC was 0.704612. Official validation ROC-AUC was 0.714429. Validation site-count slices were 576 rows at AUC 0.684695 for 0--4 sites, 122 rows at AUC 0.781654 for 5--19 sites, and 262 rows at AUC 0.758750 for 20 or more sites.

A 100-draw row bootstrap estimated validation AUC standard error at 0.016019. The final two section variants had Spearman prediction correlation 0.993803 and mean absolute prediction difference 0.013919, so they are not materially different candidates separated by this validation sample; no evaluator defect claim is supported.
