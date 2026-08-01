# Evaluation profile

## Mechanics

The protected grader runs `main.py --debug` for fast fidelity and `main.py` for full fidelity. Both modes score the full 155,845-row validation vector with RelBench's official `accuracy`, `f1`, and `roc_auc`; `fraction` and `seed` do not subsample scored rows. Validation predictions must come from a fit that never used validation labels, while test predictions may come from a separate fit over train plus validation. Test labels are absent and the hidden test metrics are unavailable.

## Input distribution

- Train contains 534,233 rows over eight semiannual origins from 2018-01-06 through 2021-07-03 and 136,183 unique papers. Origin counts rise from 506 to 136,183; per-origin positive rates range from 0.3696 to 0.4871.
- Validation contains 155,845 unique papers at 2022-01-01. Test contains 193,696 unique papers at 2023-01-01.
- At validation, paper age quantiles at 10/25/50/75/90 percent are 144/350/697/1075/1307 days. Young papers at most 182 days old are 12.64%; papers with no citation by the seed time are 31.44%.
- Mutually exclusive validation segments are 19,705 young, 34,616 older cold, 38,079 older with one or two citations, and 63,445 older with more than two citations.
- At test, 9.46% are at most 182 days old and 28.07% have no citation by the seed time. This is a measured shift toward older and more established papers relative to validation.
- Untruncated SPECTER2 token lengths for `Title [SEP] Abstract` have 25/50/75/90/95/99 percentiles 175/230/306/373/413/497. At maximum length 192, 67.08% of papers are truncated.
- Paper author counts have median 2, 90th percentile 6, and 99th percentile 19. Paper category-link counts have median 1 and 99th percentile 4. Outgoing reference counts have median 5, 90th percentile 19, and 99th percentile 44.

## Coverage axes

The implementation must cover origin time, seed age, citation coldness, multiscale citation momentum, author productivity and impact, primary and secondary categories, outgoing reference quantity and quality, matured outcomes, missing references, text length/truncation, and validation-to-test temporal shift. All dynamic database transforms are censored at each row's seed time. Synthetic labels are permitted only when their 182-day horizons close by the receiving model's cutoff.

The solution's reported roughly 35% cold/young validation share was not reproduced under a mutually exclusive definition: young plus older-cold is 34.86%, which confirms it closely. The high 67.08% token truncation rate is the main measured text limitation at length 192.

## Critical path

The score-bounding artifact is the task-adapted SPECTER2 fusion model. A bf16 batch-128 backward benchmark at sequence length 192 sustained 852 rows/second using 12.95 GB, while batch 64 with checkpointing sustained 762 rows/second using 7.25 GB. This supports three neural fits and inference within the planned 55-minute critical-path allocation.

## Strata reporting

The official harness exposes only headline metrics. The candidate records internal latest-origin AUC for the anchor, neural model, and fixed blend; no hidden test strata or labels are accessed.
