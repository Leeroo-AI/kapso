# rel-stack/post-post-related — regen diagnosis and builder study (2026-08-17)

## Why the regen fell short of the July bar (test MAP 21.78 vs banked 26.10)

The regen champion (run_0003, val 19.90) is a six-channel retrieval system
(BM25/TF-IDF/LSA lexical, tag/cluster/owner/global popularity, graph
neighbor/two-hop/PPR/SVD, BGE semantic, URL) fused with unlearned RRF plus a
gated LLM source-judgment channel. Its learned rankers were abandoned after a
real contamination finding: adjacent monthly replay folds share 56–69% of
source posts via overlapping 91-day label windows, so LambdaRank memorized
repeat sources (0.42–0.48 forward-fold MAP, 0.089 deployed on val in
run_0001). After source-disjoint purging the honest folds were tiny (107/67
groups) and the training pool thin (~500 groups); learned families lost
honestly (rrf 0.150 / linear 0.098 / lambda 0.105 weighted), so the final
recipe was RRF-dominant — leaving learned-ranking upside untested at scale.
The July winner's artifacts are lost pre-archive-era; 26.1 is not
reproducible or verifiable from any surviving artifact.

## Builder study (dev-box, one-way test scoring, campaign already finished)

Using the campaign's cached featurized groups (8 labeled monthly cutoffs,
1,824 groups, 251 pair features, ~1,500 candidates/group, 0.80 recall
ceiling):

- Honest purged rolling-origin folds: LambdaRank (min_data_in_leaf 20) beats
  RRF +0.038 weighted (0.2104 vs 0.1727); blending 0.875/0.125 with RRF
  reaches 0.2266 — every fold improved.
- Val (one-shot): my blend 0.1959 vs champion 0.1990 (tie; champion carries
  its LLM lift). 50/50 rank-average with the champion's artifact: val 0.2075.
- Test (one-way): mine 18.69; 50/50 ensemble 20.99; val-selected
  champion-heavy ensemble (w1=0.875, w2=0.35) 21.11 — all BELOW the
  champion's own 21.78.

## Lesson

Purged (source-disjoint) folds contain only unseen sources — young/cold
posts — so ranker selection on them optimizes the cold regime. The test
month skews old/warm (10.9% young sources vs 15.9% in val), and the learned
edge inverts under that shift while unlearned RRF stays robust. On this task,
fold-honest learned reranking is a val-regime win, not a test-month win;
candidate recall (not reranking) is the credible path toward the July bar.

## Outcome

The campaign harvest artifact (run_0003, test MAP 21.78) remains the best
available and the submittable vector; the builder ensembles are recorded here
as a negative result and not banked. Builder code and scored artifacts:
scratchpad `ppr_build/` (engine.py, final_grid.py, out/).
