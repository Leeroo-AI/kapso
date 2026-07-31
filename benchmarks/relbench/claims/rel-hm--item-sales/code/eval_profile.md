# Evaluation profile

## Mechanics

The immutable evaluator runs `python main.py` for full fidelity and `python main.py --debug` for fast fidelity. Both modes score all 105,542 official validation rows; `fraction` and `seed` are manifest metadata and do not change the scored rows. The score of record is the official RelBench validation MAE. R2 and RMSE are diagnostics. The evaluator checks finite, row-aligned prediction vectors and archives full-fidelity artifacts. Test labels are absent and test metrics are hidden.

Validation predictions must come from Model A, which never uses validation labels. Model B may add validation labels only for test prediction. Internal model selection uses training-only forward origins and the criterion `mean fold MAE + 0.5 * fold MAE standard deviation`.

This profile is at the repository root because the provided evaluation suite explicitly forbids creating or editing any file under `kapso_evaluation/`.

## Input distribution

- Train: 5,488,184 rows, 105,542 articles at 52 weekly Monday origins from 2019-09-09 through 2020-08-31.
- Validation: 105,542 articles at origin 2020-09-07.
- Test: 105,542 articles at origin 2020-09-14.
- Train target: mean 0.075965, median 0, 75th percentile 0, 90th percentile 0.074983, 99th percentile 1.720393, maximum 87.162593.
- Validation target metadata: mean 0.085690, median 0, 75th percentile 0, 90th percentile 0.060983, 99th percentile 1.938298, maximum 40.364915.
- Database: 15,187,287 transactions, 105,542 articles, and 1,371,980 customers. Transaction history spans 2019-09-07 through 2020-09-14.
- The seed table is a complete dense article-by-origin panel, but transactions are sparse. The cached causal weekly base contains 1,052,089 observed article-weeks and was generated in 4.082 seconds.

## Coverage axes

The evaluated rows vary by recent-sale state, four-week activity, eight-week dormancy, recent launch state, burstiness, article age, hierarchy, customer mix, price history, channel mix, and seasonal week. All three database tables contribute features.

The four training-only profiling origins are 2020-08-10, 2020-08-17, 2020-08-24, and 2020-08-31. Active-last-week counts are 19,866, 19,530, 19,932, and 19,698. Recent-four-week counts are 29,645, 29,479, 29,357, and 29,439. New-in-four-week counts are 2,684, 2,787, 3,051, and 3,307. Eight-week dormant counts are 68,912, 69,422, 69,664, and 69,721. Bursty-eight-week counts are 13,633, 13,448, 12,551, and 12,948.

## Banked deterministic measurement

The training-only forward-fold mean MAE and standard deviation were:

| Forecast | Mean MAE | Fold SD | Selection criterion |
|---|---:|---:|---:|
| Last week | 0.042800 | 0.004949 | 0.045275 |
| Last week 75% + product median 25% | 0.043157 | 0.006165 | 0.046240 |
| Median of two weeks | 0.048524 | 0.008771 | 0.052909 |
| Recency-weighted mean of four weeks | 0.052463 | 0.010517 | 0.057722 |
| Median of four weeks | 0.059303 | 0.013365 | 0.065985 |
| Median of eight weeks | 0.071122 | 0.014071 | 0.078157 |
| Zero | 0.074607 | 0.010168 | 0.079691 |

Last week wins the predeclared criterion and is the complete fallback. Its fold MAEs are 0.037987, 0.043337, 0.049460, and 0.040417.

For the zero predictor, mean absolute errors by stratum across the same folds range from 0.324217 to 0.418234 for active-last-week articles, 0.221786 to 0.288080 for recent-four-week articles, 0.693777 to 1.150998 for new-in-four-week articles, 0.004500 to 0.006912 for eight-week dormant articles, and 0.181242 to 0.342695 for bursty articles.

## Coverage reconciliation

The assumed regimes are strongly supported: about 18.5% to 18.9% of articles are active in the last week, 27.8% to 28.1% are active within four weeks, 65.3% to 66.1% are dormant for eight weeks, and 2.5% to 3.1% are recent launches. The runtime assumption is conservative for aggregation, but LightGBM fit throughput remains the score-bounding quantity and is measured at the first full forward fold.

## Full forward-fold results

The selected demand-only feature group scored 0.043920 by `mean MAE + 0.5 * SD`, versus 0.045087 for demand plus hierarchy and 0.044613 for the full customer/price widening. Their respective fold MAEs were `[0.031898, 0.038453, 0.050715, 0.038951]`, `[0.032432, 0.039898, 0.052335, 0.039079]`, and `[0.032301, 0.039742, 0.051439, 0.039080]`.

The selected 26-origin configuration scored 0.043920, versus 0.045788 for 13 origins and 0.044636 for 52 origins. The frozen fold-median tree count was 453.

A predeclared post-fit comparison used reconstructed training-only OOF predictions. The 50% model plus 50% last-week fallback blend scored 0.042884 with fold MAEs `[0.033670, 0.039863, 0.048556, 0.036550]`, compared with 0.043940 for the model alone and 0.045275 for the fallback. It improved two folds and lowered fold variance, so it was retained for the weekly-only stage before daily widening superseded it.

## Daily trajectory widening

Seventy-seven strictly causal daily sales, transaction, distinct-customer, channel, price, activity-day, and daily-recency features were added to the retained weekly demand block. They improved all four folds from `[0.031898, 0.038453, 0.050715, 0.038951]` to `[0.030153, 0.034873, 0.045345, 0.035243]`; the selection criterion improved from 0.043920 to 0.039601. Frozen rounds are the fold median, 682.

Daily-model stratum MAEs by origin were:

| Origin | Active | Recent four weeks | New four weeks | Dormant eight weeks | Bursty eight weeks |
|---|---:|---:|---:|---:|---:|
| 2020-08-10 | 0.137509 | 0.096382 | 0.360720 | 0.004501 | 0.096244 |
| 2020-08-17 | 0.160121 | 0.110324 | 0.441900 | 0.005880 | 0.113358 |
| 2020-08-24 | 0.207595 | 0.144912 | 0.585675 | 0.006912 | 0.177907 |
| 2020-08-31 | 0.159169 | 0.110204 | 0.429833 | 0.006542 | 0.145400 |

A repeated fallback blend grid selected 75% daily model by a criterion margin of only 0.000006 while worsening mean fold MAE. This is within noise, so the simpler unblended daily model is retained.
