# Evaluation profile

## Mechanics

- The immutable registered entrypoint delegates to `grader.py`, launches `main.py` in an isolated subprocess, validates both prediction arrays, evaluates the full official validation table, and archives full-fidelity outputs.
- The primary score is official validation ROC-AUC. Accuracy, average precision, F1, and ROC-AUC are all emitted. `--fraction` and `--seed` are manifest metadata and do not change the scored rows.
- Model A must produce validation predictions without validation-label fitting. Model B may add closed validation outcomes for test prediction. The test labels are absent.

## Input distribution

- Train: 19,239 rows, 8,517 users, 22 weekly timestamps from 2012-06-20 through 2012-11-14, 3,247 positives, rate 0.16877. Rows per origin range from 7 to 3,108; the last four origin rates are 0.18775, 0.10650, 0.09807, and 0.10037.
- Validation: 2,013 unique users at the single origin 2012-11-21, 223 positives, rate 0.11078.
- Test: 1,958 unique users at the single origin 2012-11-29.
- Daily reconstruction from 2012-06-20 through 2012-11-21 yields 144,271 user-origin rows and 23,554 positives before chain-specific cutoff.
- Validation users: 30.15% have zero prior attendee rows, 6.26% have zero resolved outgoing neighbors, and 0.05% have zero full friendship degree. Test counterparts are 27.68%, 5.11%, and 0.00%.
- Full friendship scan: 30,386,403 rows; 213,703 resolved directed pairs, 29,778,857 user-only rows, 1,741 friend-only rows, and 392,102 rows with neither endpoint. The deduplicated symmetric graph has 216,902 nonzero adjacency entries.
- Known-user attendee history contains 49,822 rows over 9,257 users. Invitation status accounts for 39,430 known-user rows. Interests contain 14,978 rows over 1,979 users.
- The user table has 64 locales, dominated by `en_US` and `id_ID`. Fewer than 0.2% of validation or test rows belong to locales with at most ten database users.

## Coverage axes

- Historical attendee exposure: zero, sparse, and rich invitation history.
- Graph availability: isolated in the resolved graph, partially resolved, and high-coverage friendship rows.
- Locale prevalence and demographic availability.
- Temporal origin and changing label/event volume.
- Interest availability, event blast exposure, and closed historical positive outcomes.

## Solution-claim check

- The measured 30.2% zero-history validation claim is reproduced at 30.15%.
- Resolved graph availability is slightly lower than the rounded claim: 213,703 directed resolved rows rather than about 217K. Explicit isolated fallbacks remain necessary.
- Validation is a single time slice with a materially lower rate than much of earlier training history, so all design selection will use purged forward origins and validation will only be reported once.

## Critical path probe

- Loading the sanitized database takes about 4.5 seconds. Null-pattern scan plus full-degree aggregation takes about 0.4 seconds after load.
- Naive exact daily label construction produces 144,271 rows in 11.4 seconds, or 12,655 rows/s. Two sparse propagation steps across all 37,143 users run at about 863 seeds/s.
- The bounded artifact is therefore the wide censored per-seed feature panel and its purged model gates, not CSR construction or SVD multiplication.

## Full-run measurement

- The full temporal matrix covers 156 unique feature origins and builds in 55.2 seconds, about 169 origins/minute. Purged forward gating takes 30.3 seconds and the two final expert chains take 6.6 seconds.
- Four purged forward history AUCs are 0.81491, 0.87862, 0.92952, and 0.94025. The selected 16-dimensional, hard-zero graph mixture scores 0.82958, 0.87771, 0.92997, and 0.93843, improving equal-fold mean AUC from 0.89082 to 0.89392.
- The 32-dimensional comparator and community variants did not pass the internal selection gate. The selected full Model A validation ROC-AUC is 0.84481.
- Validation slice AUCs are 0.70737 for zero history (607 rows), 0.88777 for nonzero history (1,406), 0.88595 for resolved-graph isolates (126), 0.84322 for resolved-neighbor users (1,887), 0.92700 for low friend coverage (1,070), and 0.74427 for high friend coverage (943).
- The 100-draw bootstrap standard error of validation AUC is 0.01666. History, cold, and mixture validation predictions have mean pairwise Spearman correlation 0.93787 and AUCs 0.83854, 0.83562, and 0.84481 respectively.
- The zero-history label rate is highly nonstationary even when total origin volume is similar: 9/1,056 (0.85%) at 2012-11-07 and 6/730 (0.82%) at 2012-11-14 versus 30/607 (4.94%) in validation. Overall rates are much closer (9.81%, 10.04%, and 11.08%), so the aggregate masks a conditional cold-user regime change capable of reversing graph-specialist rankings.
- A 22-origin daily diagnostic is intentionally not used as the model-selection estimate because adjacent seven-day labels overlap. It gives history mean AUC 0.89845 versus best graph mixture 0.89830, whereas the four non-overlapping weekly origins give 0.89366 versus 0.89689 after the compact/sparse-weight refinements.
- The refined Model A validation run scores history 0.84070, cold 0.83652, and mixture 0.83835. Its 100-draw bootstrap AUC standard error is 0.01735, while pairwise prediction Spearman correlations are 0.90012, 0.94914, and 0.95774; all candidate score gaps are below two standard errors.
- Registered full run `run_0006` scored accuracy 0.92101, average precision 0.61924, F1 0.56911, and ROC-AUC 0.83835 over all 2,013 validation rows.
