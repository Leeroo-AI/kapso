# Evaluation profile

The protected evaluator invokes `main.py` once for every rolling origin, assembles tick-level test vectors into official row order, then computes all metrics on 588 validation rows. The search score is global validation ROC AUC, so cross-origin positive-negative pairs dominate and raw within-field percentiles are not sufficient. Full fidelity uses the complete 27 validation and 30 prediction origins; `fraction` does not subsample scoring rows.

## Input distribution

| Split | Rows | Origins | Field size min / median / max | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,353 | 59 | 20 / 22 / 32 | 0.1707 |
| Validation | 588 | 27 | 20 / 22 / 24 | 0.2024 |
| Prediction | 726 | 30 | 22 / 24 / 26 | unavailable |

The validation origins span 2005-03-02 through 2008-03-16 and prediction origins span 2010-03-02 through 2013-03-16. Rolling snapshots contain 1,353 to 2,645 closed exact-window labels and grow from 18,469 to 21,715 results and from 2,228 to 5,466 qualifying rows. The prediction field is measurably larger than validation.

## Coverage axes

- Origin year and season phase, including long winter gaps.
- Field size and within-origin competitive rank.
- Driver history depth: validation none/sparse/rich counts are 20/23/545; prediction counts measured from the legal tick snapshots are 17/13/696.
- Driver recency: validation none/>365d/92-365d/<92d counts are 20/7/75/486; prediction counts are 17/9/60/640.
- Exact qualifying availability versus grid-derived historical sessions.
- Constructor continuity, teammate comparison, standing position, recent form, and expected sessions inside the label window.

The static 2009 database view misleadingly classifies 28.2% of prediction rows as zero-history because it omits rolling history. The legal per-origin snapshots reduce this to 2.3%, so the solution's rolling-safe loader is essential.

## Internal selection and strata

The prescribed expanding yearly tests contain 66/22/42/183/187 rows for 2000 through 2004, with positive rates 0.1667/0.1364/0.1429/0.2022/0.2406. Model selection uses these forward folds only. Evaluation reporting should include yearly fold AUC plus history-depth and recency strata wherever both classes occur.

This profile is stored outside `kapso_evaluation/` because the task explicitly marks every file under that directory immutable and says any edit there voids the score.

## First-iteration diagnostics

Run `run_0001` scored ROC AUC 0.9117736647. A 200-draw row bootstrap estimated standard error 0.01305484 and a 95% percentile interval of [0.88668075, 0.93703885]. The gated stacked alternative scored 0.91209618 only for this diagnostic, a +0.00032252 difference, and had Spearman rank correlation 0.97700126 with the archived classifier. These candidates are not sufficiently different and are nowhere near separated by two standard errors, so the resolution check does not establish an evaluator defect.

Validation AUC by origin year was 0.85874 (2005, n=165), 0.88207 (2006, n=178), 0.97763 (2007, n=201), and 0.86806 (2008, n=44). By history depth it was 0.21053 for none (n=20), 1.0 for sparse (n=23), and 0.90938 for rich (n=545); the none and sparse slices each contain only one positive. By recency it was 0.69321 for 92–365 days (n=75) and 0.93883 for under 92 days (n=486); the >365-day slice had no positives.

The 29 legally closed prediction-era origins available in the final rolling training snapshot contain 704 rows, mean field size 24.276, and label rate 0.17330. Validation contains 588 rows across 27 origins, mean field size 21.778, and label rate 0.20238. This shift is recorded but does not by itself establish inverted ROC AUC candidate ordering, so no evaluation-change request is warranted.

After the first-iteration diagnostic, a feature compactness decision was made solely from the prescribed 2000–2004 forward folds. Removing raw numeric circuit-cluster identity and two nationality cohort scalars while retaining recent circuit-cluster performance raised weighted forward AUC from 0.89994 to 0.90364 and worst-fold AUC from 0.84691 to 0.85061. Full run `run_0002` then scored ROC AUC 0.9128845568; this validation result was not used to select or revise the candidate.

## Constructor-season prequential profile

The frozen iteration-2 replay uses pseudo-test years 2001–2004. For each year Y, its base fit contains only exact windows closed by December 31 of Y-3; earlier same-year pseudo-test origins enter the fit only after their 30-day windows close. The full replay contains 541 rows across 28 origins, including 107 synthetic pre-opener rows. Whole-origin bootstrap comparisons use 2,000 draws.

The compact identical-origin baseline scored pooled AUC 0.87735, mean yearly AUC 0.89383, and worst-year AUC 0.84322. Its recency AUCs were 0.85113 for `<92d` (n=321), 0.90741 for `92–365d` (n=193), and 1.0 for `>365d/no-history` (n=27). Opener-related rows account for 154 of the 193 medium-recency rows, or 79.79%; opener concentration is substantial but not near-total, correcting the solution's unverified assumption.

No constructor-season prefix passed its frozen adoption gate. Prefix 1 changed mean/worst/medium-recency AUC by +0.00062/-0.00191/+0.00307 with 20.3% positive bootstrap support. Prefix 2 changed them by +0.00007/-0.00708/-0.00217 with 4.5% support. Prefix 3 changed them by +0.00542/-0.01470/-0.01971 with 0% support. The deployed prefix is therefore 0; the exact synthetic origins, one-origin-total weighting, eight-year half-life, and 20% opener-mass cap remain part of the prescribed training distribution.
