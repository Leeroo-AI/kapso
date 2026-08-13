# Evaluation profile

## Mechanics

The immutable registered evaluator invokes `main.py` and scores the complete 409,792-row validation vector; `fraction` does not subsample rows. It requires finite float probabilities aligned to the original validation and 351,885-row test tables. Validation ROC-AUC is the score of record. Model A must exclude validation labels from its complete chain; Model B may use validation labels for test predictions.

## Input distribution

- Train has 4,708,383 rows, 1,486,748 distinct customers, 31 quarterly origins from 2008-01-10 through 2015-07-02, and churn prevalence 0.623957. Origin sizes span 5,372 to 416,981 and prevalence spans 0.571010 to 0.743591.
- Validation is one origin at 2015-10-01 with 409,792 distinct customers and prevalence 0.642028. Test is one origin at 2016-01-01 with 351,885 distinct customers.
- The database has 12,644,508 reviews from 1,584,084 customers and 416,125 observed products. Rating mean is 4.3440, verified share is 0.7043, mean review-text length is 572.57, and text and summary are almost complete.
- Product price median is 12.99. Category and description null rates are 3.73% and 7.19%; brand and title are complete.
- Every seed is selected by a review in the immediately preceding 91-day window, so seed-history depth is at least one event while earlier history remains long-tailed.

## Reliability

Across training origins, variance of origin churn means is 0.00313735 versus mean binomial variance 0.000004324, a ratio of 725.6. The 91-day review volume falls 15.56% from the pre-validation window to the pre-test window. Origin drift, rather than iid sample noise, is the dominant model-selection risk; the expanding origin 20/24/30 design and cutoff-specific pretraining states are therefore required.

The evaluator supplies only a single validation origin. Candidate acceptance uses internal forward-origin predictions, customer-clustered paired bootstrap support, and median improvement rather than official validation feedback. The reproduced champion's official validation bootstrap SE is 0.000805. The admitted 0.2 transformer blend improved forward origins by +0.000303/+0.000194/+0.000813 with paired supports 0.98/0.97/1.00 and then measured 0.712039 officially. Representativeness is evaluated from training-era origin prevalence and event volume only.

## Coverage axes

Coverage axes are origin/regime, event-day history depth and truncation, recency, recent activity 1/2-3/4+, completed gap behavior, rating and verification, review semantic state, product/category/brand rarity, product popularity at event time, price and missing metadata, calendar season, and system activity at each origin.

## Critical path

The score-bounding artifact is the family of cutoff-specific self-supervised trunk checkpoints. The first measured training checkpoint determines achievable sequences per second and whether two chronological epochs fit before freeze. Champion predictions and tabular features are consumers and ensemble partners, not the bounded artifact.

## Iteration 2 extension

The immutable evaluator and row distribution are unchanged. The rescore of archived run_0014 used all 409,792 validation rows and returned ROC-AUC 0.7123846047, average precision 0.7914661750, accuracy 0.6330284632, and F1 0.6786668205. Its validation and test files were banked with SHA-256 values `4b89f996a5f2960b1bc1c5119f42352778116b8f20a80393155fa527aa6abde3` and `4906c7d82c690d8ae936f033352a8f861a9f304dbfb155d038ac4dba7f8de8fb`.

The measured sparse axes are prior-label depth 0 / 1-2 / 3+, recent activity 1 / 2-3 / 4+, and recency 0-14 / 15-30 / 31-60 / 61-91 days. Existing forward OOF slices have AUC 0.6493 for no prior label, 0.6199 for one recent review, and 0.6493 for recency 30-91, so the new measurement reports pooled and sparse-slice movement separately. Shared dependency verification found 12,644,508 ordered review edges, 4,708,383 lane-1 tabular and semantic seed rows, MiniLM arrays shaped 4,708,383/409,792/351,885 by 96, and ordered semantic arrays shaped 4,708,383/409,792/351,885 by 36.

For this iteration the score-bounding artifact changes to causal residual-graph and external-customer retrieval predictions at origins 20, 24, and 30 plus validation/test. The precommitted rates are above 500,000 graph seed rows per minute and above 500 retrieval queries per second, with a hard 90-minute retrieval projection cap.

The measured graph rate was 7,374,732 seed rows/minute. CPU IVFPQ retrieval measured 37,553 queries/second on the fold-20 benchmark and projected 43.9 seconds for all 1,648,963 forward/final queries, so no retrieval reduction was required. The kNN channel inverted between development origins and was excluded; the 0.10 residual-graph rank contribution moved origin 30 by +0.0000125 with 1,000-draw customer-clustered SE 0.0000135 and P(improvement)=0.806. Activity-1 moved +0.0000070 while activity 2-3 moved -0.0000147; recency 0-14 / 31-60 / 61-91 moved +0.0000219 / +0.0000248 / +0.0000133.

Registered full-fidelity run_0017 scored all 409,792 validation rows at ROC-AUC 0.7124067197, average precision 0.7914916751, accuracy 0.6332139232, and F1 0.6789198565. Relative to banked run_0014, the official validation diagnostic was +0.0000221, far below the 0.0007645 row-bootstrap SE; prediction rank correlation was 0.999886. The official check is post-freeze evidence only and did not participate in feature or weight selection.
