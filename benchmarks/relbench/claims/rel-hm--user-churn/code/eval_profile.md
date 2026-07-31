# Evaluation profile

## Measurement mechanics

The immutable registered entrypoint runs `main.py` in a child process, requires `val_predictions.npy` and `test_predictions.npy`, computes all official validation metrics through `task.evaluate`, and uses validation `roc_auc` directly as the manifest score. Full fidelity does not subsample items; `--fraction` and `--seed` are manifest metadata. Test labels are absent, and only validation prediction shape/finiteness receives a direct grader check, so the candidate also runs the provided checker on both arrays.

Validation predictions must come from Model A trained without validation labels. Model B may use train plus validation labels only for test prediction. Features are point-in-time: a seed at `t` may use transactions through `t`, while historical outcomes and encodings require their outcome window to end by `t`.

## Input distribution

- Train: 3,832,692 rows, 983,616 customers, 52 Monday timestamps from 2019-09-09 through 2020-08-31.
- Validation: 76,556 customers at 2020-09-07, positive rate 0.812804.
- Test: 74,575 customers at 2020-09-14.
- Train row counts per timestamp range from 30,797 to 136,619; timestamp positive rates range from 0.7245 to 0.8882, showing strong seasonal and regime variation.
- Every train, validation, and test seed has at least one transaction in its required `(t-7d,t]` conditioning window.
- Distinct-customer overlaps are 68,574 train-validation, 66,264 train-test, and 14,331 validation-test.
- The transaction table has 15,187,287 rows, 999,345 customers, 71,375 sold articles, and 374 purchase dates.
- Customer-day baskets average 3.41 rows; row-count quantiles at 50/75/90/95/99 percent are 2/4/7/10/17.
- There are 3,983,823 active customer-weeks. Weekly baskets average 3.81 rows, 1.12 active days, 3.40 distinct articles, and spend 0.108 in stored price units.
- Train customers have a median of 2 active weeks; 90/95/99 percent quantiles are 9/12/20.
- Among train seeds, coverage of at least 2/3/4/8 active weeks is 74.3/57.9/45.7/18.7 percent. Previous-active-week gap quantiles at 50/75/90/95/99 percent are 4/8/16/22/34 weeks.
- Coverage of at least 2/3/5/10 active purchase days is 75.9/60.4/39.8/15.8 percent.
- Article cardinalities for product code/type/group, department, index group, section, garment group, color, and perceived master color are 47,224/132/19/299/5/57/21/50/20.

## Coverage axes

The implementation must cover seed date/regime, customer history depth, renewal gap and cadence, current basket volume and composition, channel behavior, customer demographics/membership/postal frequency, article/category diversity and novelty, category-specific continuation priors, paid-price affinity, article popularity momentum, and global market activity. Sparse-history customers require smoothed priors and explicit missing/coverage indicators. The assumed weekly cadence is confirmed exactly; the previously reported 0.7119 AUC remains unverified until a training-only forward holdout is banked.

## Slice reporting

The registered harness exposes only the complete validation aggregate. Training-only forward-fold diagnostics will additionally report counts and AUC by history-depth, basket-size, recency-gap, and age strata without using official validation scores for selection.

## First full measurement

The banked core forward probe reached 0.70860 AUC, so the assumed 0.7119 probe was not reproduced. Four recent forward folds scored 0.71905/0.71926/0.70980/0.70783 for the retained core. The broad all-table widening improved mean fold AUC by about 0.0011, below the precommitted 0.002 tie threshold, while CatBoost and XGBoost OOF-rank blends added only 0.00029 and 0.00020; all were rejected. Official aggregate validation metrics were average precision 0.905380, accuracy 0.812804, F1 0.896737, and ROC-AUC 0.718012.

History-depth slice AUC was weakest for 2–3 and 4–7 prior active weeks (0.6041 and 0.6050), while current baskets of 8+ rows reached 0.7340. The next feature iteration therefore targets renewal-survival estimates and causal activity-state/category interactions rather than another tree-family change.

## Second full measurement

The compact priority block improved every forward fold. Its regularized 63-leaf candidate scored 0.72011/0.72074/0.71144/0.70921 versus core 0.71905/0.71926/0.70980/0.70783, a stable mean gain near 0.0014. The fixed 0.002 tie rule rejected it and reproduced the identical official 0.718012 score. Because the solution explicitly permits resolving sub-0.002 ties with bootstrap uncertainty, the next selection computes a deterministic paired bootstrap on the complete training OOF predictions and retains a sub-threshold block only when its 95% lower bound is positive.

## Third full measurement

The paired OOF bootstrap estimated the priority block gain at 0.00140 with a 95% interval of [0.00085, 0.00203], so the block was retained. Official validation ROC-AUC improved to 0.719120 and average precision to 0.905872. History-depth slice AUC improved from 0.6041 to 0.6111 for 2–3 active weeks, from 0.6050 to 0.6076 for 4–7, and from 0.6219 to 0.6266 for a single active week. CatBoost/XGBoost still added only about 0.00025/0.00026 in OOF rank blends and were rejected.
