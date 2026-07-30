# Evaluation Profile

The immutable grader runs `main.py` as an isolated child, requires finite float vectors of shapes `(1766,)` and `(1816,)` in original task-row order, and scores every validation row. The primary score is validation MAE; R2 and RMSE are secondary. Full fidelity does not subsample. The harness controls fidelity, output location, child timeout, and the manifest's fraction/seed fields; it does not choose model settings, post-processing, or inference context.

The measured split sizes and origins are train 5,100 rows at April 26/30 and May 4 (1,535/1,530/2,035), validation 1,766 rows at May 8, and test 1,816 rows at May 14. Historical-impression volume counts for train are 818/637/1,781/1,604/260 in bands 0, 1–10, 11–100, 101–1,000, and 1,000+; validation counts are 99/96/491/795/285; test counts are 101/71/410/867/367. The later splits therefore contain proportionally fewer cold rows and more 101+ history rows than train.

Coverage axes are replay origin, availability depth, the five history-volume bands, category hierarchy, price band, title morphology and character SVD, HistCTR/position moments, audience mixture, empirical-Bayes cohort priors, and Visit/Phone cohort intensity. Train/validation/test cover 31/30/31 categories. Visit cohort features are nonzero for 5,098/1,764/1,815 rows and Phone cohort features for 5,076/1,763/1,815 rows. Direct Visit/Phone seed-ad overlap is zero, so those tables contribute only through censored cohort paths.

The common cached three-fold OOF has 5,454 rows at May 1/2/4 (1,631/1,788/2,035), with volume-band counts 552/491/1,777/2,212/422. This iteration extends the origin axis to five daily folds from April 30 through May 4. Exact replay provides 15,222 eligible model-A labels through May 4 and 26,646 model-B labels through May 10.

The score-bound artifact is the five-fold TabPFN-v2 OOF matrix because the frozen champion and 406-column causal feature matrix already exist. Candidate weights are selected only from replay OOF labels using leave-one-origin-out meta-cross-fitting and a 5,000-replicate day-block bootstrap; official validation labels are reserved exclusively for model B's test-producing fit.
