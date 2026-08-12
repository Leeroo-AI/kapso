# Evaluation profile

## Mechanics

The immutable registered evaluator runs `main.py` in an isolated subprocess, validates complete aligned validation and test NumPy arrays, and passes validation predictions to the official RelBench evaluator. Both fast and full fidelity score all 247,398 validation rows. The search score is validation ROC AUC; full runs archive both prediction arrays. Model A must produce validation predictions without any validation-label fit, while Model B may refit on train plus validation exclusively for test predictions.

## Input distribution

Training has 3,386,276 rows over 40 quarterly origins, validation has 247,398 rows at 2020-10-01, and test has 255,360 rows at 2021-01-01. Positive rates are 4.81% overall training and 2.95% validation. At validation, the authored-activity segments are never-active 160,847 rows and 1,083 positives, dormant over one year 70,120 and 3,645, stale 92–365 days 11,409 and 1,302, and active under 92 days 5,022 and 1,271. Thus 93.3% of rows and 64.8% of positives are never-active or dormant.

The relational database has 163,748 tagged questions among 333,893 posts, 1,175,368 post-history records, 1,317,876 votes, 623,967 comments, and 77,337 links. Current post title and tag values are mutable, so earlier origins require state reconstruction from post-history types 1/4 and 3/6. Votes lack usable voter identity but retain post identity, allowing cutoff-valid inbound topic traffic.

## Coverage axes

- Origin and platform regime.
- No-history, dormant, stale, and active user state.
- Question age, old-question ownership, and owned-question depth.
- Topic momentum at 30, 91, and 365 days and corresponding prior-year windows.
- Inbound votes, comments, answers, and links routed to root questions.
- Historical title/tag state availability, text script, and missing reconstruction.
- Content-derived future-traffic predictions versus deterministic tag traffic.

## Metric diagnostics

Feature admission uses three purged forward training folds, paired fold AUC changes, and dormant-slice stability. The official validation metric is reported only after the design is frozen. The candidate records a 100-draw bootstrap standard error and prediction rank correlations, and reports activity-density and dormant-recency slices wherever both labels occur.

The accepted tag block changed the three fold AUCs by -0.000027, +0.000621, and +0.000203. The content-traffic block changed them by +0.000673, +0.001582, and +0.000260. Together they changed them by +0.000424, +0.002328, and +0.000259, with a +0.001004 mean overall improvement and +0.002938 mean dormant-slice improvement.

Registered full run_0006 scored validation ROC AUC 0.9047596371, average precision 0.3616787516, accuracy 0.9711638736, and F1 0.2785194175. Slice results were: never-active 160,847 rows, label rate 0.006733, AUC 0.819983; dormant 70,131 rows, rate 0.051988, AUC 0.830789; stale 11,395 rows, rate 0.114173, AUC 0.827095; active 5,025 rows, rate 0.252935, AUC 0.838489.

The candidate-versus-reproduced-champion paired bootstrap delta was +0.0007535 with SE 0.0002075, 95% interval [0.0004412, 0.0011913], and P(delta > 0)=1.00. Their rowwise Spearman correlation was 0.995601. The decorrelated solo content expert had correlation 0.635512 but ROC AUC 0.801238, outside the two-SE finalist range, so there is no unresolved resolution defect or qualified cross-branch ensemble. Trailing 91-day event volume was 123,048 at validation and 116,304 at test; the 5.5% decline is consistent with adjacent history rather than an isolated validation shock.

## Coverage check

The solution's 28.3% dormant-validation share, approximately half of positives in the dormant segment, and full non-null current tag coverage are consistent with the measured campaign profile. The assumed English-language dominance is checked through text script fractions; non-Latin or unreconstructable states route through tag/numeric features and missingness flags instead of unsafe current text.

## Iteration 2 cold-start coverage

The requested replay system targets two strict never-authored strata: `N0` has no authored posts, comments, attributable post-history, or prior badge; `N1` has no authored activity but at least one prior badge. Monthly landmarks vary by origin regime, account-age decile, prior-badge depth, creation cohort, ID/account rank, badge family, and system incidence. The final gate reports global, never-authored, N0, and N1 deltas across six purged quarterly origins and uses UserId-clustered paired resampling.

The debug profile built 12 monthly origins into 127,567 sampled rows, 14,178 positives, and 204 causal features in 23.9 seconds. Because those latest 12 replay origins cannot close before the first four of the six diagnostic folds, debug treats missing experts as constants and is only a wiring check; the full 84-origin run is the promotion measurement.
