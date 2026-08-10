# Evaluation profile

The immutable evaluator launches `main.py` in an isolated child, requires positionally aligned finite probability vectors of shapes `(29979,)` and `(36129,)`, and computes all official metrics over the full validation split. `--fraction` is manifest metadata only. The score of record is validation ROC-AUC; test labels are absent.

## Input distribution

| split/origin | rows | unique users | positive rate |
|---|---:|---:|---:|
| train 2015-04-26 | 29,210 | 29,210 | 0.9051 |
| train 2015-04-30 | 25,007 | 25,007 | 0.9013 |
| train 2015-05-04 | 32,402 | 32,402 | 0.9101 |
| validation 2015-05-08 | 29,979 | 29,979 | 0.9035 |
| test 2015-05-14 | 36,129 | 36,129 | unavailable |

Daily reconstructed origins from April 26 through May 10 contain 452,999 rows. Daily positive rates range from 0.8841 to 0.9113 and populations range from 24,140 to 40,510. Exact-label checks at April 26, April 30, and May 4 matched official sorted UserIDs and labels after excluding VisitStream rows with null UserID; an unrestricted aggregation has one extra null-user group per origin.

Event volume varies materially by day. Visit volume ranges from 195,766 to 363,592 on complete days; search impressions from 271,025 to 538,005; searches from 76,447 to 149,442; phones from 7,376 to 18,814. May 14 contains only six visit rows because it is the input cutoff boundary.

## Coverage axes

- Origin date and available-history length.
- Sparse, medium, and heavy prior visit activity; no-history/cold users.
- Prior search and phone presence.
- Seen versus unseen users and device/hierarchy identifiers.
- Category, parent-category, subcategory, location, region, city, ad, and IP breadth/concentration.
- Session depth/recency, channel funnels, activity momentum, and within-origin relative standing.

The solution assumption about regenerated populations was confirmed exactly at all three official train origins. The critical-path benchmark aggregated core visit features for 452,999 seed rows in 1.69 seconds, or 268,587 seed rows/second, excluding downstream feature widening and model fitting.

## Final internal evaluation

The selected v15 LightGBM representation achieved equal-origin purged-forward mean ROC-AUC 0.701883 and worst-origin 0.683868 over April 30, May 2, and May 4. UserID-bootstrap SE on pooled rows was 0.003298. Slice ROC-AUC/counts were: cold/no-visit 0.511588/21,475; sparse 0.551096/8,754; heavy 0.638349/16,204; seen 0.724415/62,596; unseen 0.519579/23,282; phone 0.742419/31,754; no-phone 0.636802/54,124; search 0.722208/47,887; no-search 0.606509/37,991.

The registered full-fidelity run_0018 scored validation ROC-AUC 0.713714. The surrounding validation origin has label rate 0.9035 and ordinary event volume relative to adjacent days, so no representativeness defect was observed.
