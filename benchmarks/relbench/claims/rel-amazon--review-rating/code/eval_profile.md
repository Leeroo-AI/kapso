# Evaluation profile

## Measurement mechanics

The immutable grader runs `main.py` in an isolated child, scores all 806,355 validation rows with official RelBench R2, MAE, and RMSE, and uses R2 as the untransformed score of record. `--fraction` and `--seed` are manifest metadata only for this suite. Validation predictions must come from a chain fit only on training labels; the test chain may refit on training plus validation labels. Full runs archive both complete prediction arrays.

The profile is kept outside `kapso_evaluation/` because the registered evaluation explicitly declares every file under that directory immutable.

## Input distribution

| Split | Rows | Time range | Users | Products | Verified | Mean target |
|---|---:|---|---:|---:|---:|---:|
| Train | 11,822,796 | 2008-01-02 to 2015-09-30 | 1,726,409 | 232,496 | 72.07% | 4.3390 |
| Validation | 806,355 | 2015-10-02 to 2015-12-31 | 511,466 | 80,423 | 71.13% | 4.4162 |
| Test | 8,217,532 | 2016-01-02 to 2018-09-28 | 1,289,066 | 422,343 | 68.00% | unavailable |

Training target counts for ratings 1 through 5 are 373,738; 472,839; 1,112,734; 2,675,667; and 7,187,818. Validation counts are 21,799; 27,206; 66,780; 168,399; and 522,171. The validation target standard deviation is 0.9672.

Relative to training, validation has 1.64% cold-user rows and 2.24% cold-product rows. Test has 3.75% cold-user and 44.99% cold-product rows relative to training; after adding validation to chain B these are 3.34% and 44.77%. This confirms that content-generated product factors are mainly a long-horizon test requirement rather than a validation lever.

Validation user-history row counts by prior-count buckets 0, 1, 2, 3-5, 6-10, 11-25, 26-100, and over 100 are 13,211; 34,141; 56,522; 180,321; 151,910; 156,995; 131,571; and 81,684. Corresponding product buckets are 18,099; 5,529; 5,335; 17,007; 26,623; 64,103; 182,808; and 486,851.

Product metadata has 3.73% null category, 7.19% null description, no null title/brand/price, 239,719 brands, and median price 12.99. Customer names are 0.0036% null with 1,216,669 distinct strings. The sanitized review projection contains exactly primary key, timestamp, customer ID, product ID, and verified flag.

## Coverage axes

- Forecast horizon: three-month validation-like episodes and 24-33-month long-lag episodes.
- History geometry: cold, one, two, 3-5, 6-10, 11-25, 26-100, and over 100 prior interactions.
- Entity novelty: warm/cold user, warm/cold product, both cold, and repeat pair.
- Staleness and drift: recency since prior user/item interaction, signed damped time deviation, calendar month, weekday, and elapsed time.
- Metadata: category path, brand frequency, title/description embedding, log price, verified, and customer-name morphology.
- Output: continuous clipped conditional expectation, reported overall and by history/cold/horizon strata on internal forward folds.

## Solution discrepancy check

The measured profile agrees with the supplied coverage claims: validation is short and warm while test is long and approximately 45% cold-product. The assumed value of unlabeled post-cutoff implicit updates must be checked against frozen histories on the internal long-lag fold; no official validation labels are used for that decision.

