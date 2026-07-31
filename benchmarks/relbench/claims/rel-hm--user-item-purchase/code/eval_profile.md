# Evaluation profile

## Mechanics

- Registered full evaluation runs `main.py` once with an 28,800-second timeout and scores all 74,575 validation rows.
- `fraction` and `seed` are manifest metadata only; neither changes the scored rows.
- The primary metric is official `link_prediction_map` at 12. Precision and recall at 12 are also reported.
- Predictions must have shape `(74575, 12)` and `(67144, 12)`, integer dtype, legal article indices, distinct entries per row, and original row order.
- Model A validation predictions cannot use validation labels. Model B alone may continue on train plus validation labels for test predictions.

## Input distribution

- Train has 3,878,451 positive customer-week rows over 52 Monday origins from 2019-09-09 through 2020-08-31.
- Validation has 74,575 rows at 2020-09-07; test has 67,144 rows at 2020-09-14.
- Train target-set size has mean 3.40, median 2, 90th percentile 7, 99th percentile 16, and maximum 161.
- Validation target-set size has mean 3.18, median 2, 90th percentile 7, 99th percentile 15, and maximum 47.
- The database has 15,187,287 transactions, 999,345 observed customers, and 71,375 observed articles across 374 days. Channel 2 contributes 10,903,537 transactions and channel 1 contributes 4,283,750.
- Among validation seed customers with prior history, median transaction count is 22 and median distinct-item count is 20. The history-count bins `1-5`, `6-10`, `11-20`, `21-50`, `51-100`, and `101+` contain 9,058, 8,620, 13,964, 21,190, 9,998, and 3,998 rows; 7,747 rows have no prior transaction in the database.
- Among validation seed customers with prior history, median recency is 28 days. Recency bins `0-1`, `2-3`, `4-7`, `8-14`, `15-28`, `29-90`, and `91+` contain 4,959, 4,511, 6,195, 8,304, 9,855, 21,404, and 11,600 rows.
- Validation targets are 3.52% exact repeats and 7.95% from a previously observed product code; 9.0% of rows have any exact repeat. This contradicts an interpretation that repeat purchase is the dominant route and makes explore/trend retrieval the critical channel.
- At the validation cutoff, the trailing 7 days contain 266,302 transactions, 76,556 customers, and 19,573 articles. The top 12 articles account for only 2.83% of these transactions.

## Coverage axes

- Temporal origin and short-window trend regime.
- Target-set cardinality.
- New customer versus observed customer.
- History count and distinct-item count.
- Days since last purchase.
- Exact repeat, known-product sibling, and catalog exploration.
- Dominant sales channel, age cohort, club/news metadata, and postal-frequency bucket.
- Product hierarchy, description/name hashes, price compatibility, velocity, and transitions.

## Initial train-only measurements

- Global popularity MAP@12 across the final three training origins ranges from 0.00560 to 0.00816 for trailing 7 days and from 0.00617 to 0.00743 for trailing 3 days.
- Three-layer convergence and frequency-enriched attention remain assumptions to be checked by throughput and recent forward-origin retrieval measurements.

## Stratum reporting plan

- Record count and MAP@12 by history-count bin, history-recency bin, target-set-size bin, and repeat-versus-novel target stratum when predictions are banked.
