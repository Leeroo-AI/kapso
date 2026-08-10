# Evaluation profile

## Mechanics

The immutable registered evaluator runs `main.py` in a subprocess, validates full-length aligned NumPy predictions, and calls the official RelBench task evaluator. Fast fidelity changes the candidate build to `--debug` but does not subsample scoring rows. The score of record is validation ROC AUC; all full runs archive validation and test predictions. Validation predictions must come from a chain fitted without validation labels, while the test chain may refit supervised components on train plus validation.

## Input distribution

The training table contains 3,386,276 rows across 40 quarterly origins from 2010-10-14 through 2020-07-02 and 239,945 distinct users. Snapshot size grows from 1,093 to 239,945. The validation table is one 247,398-user origin at 2020-10-01, and test is one 255,360-user origin at 2021-01-01. Training positive rate declines with platform age: 27.8% at the first origin, 2.39% at the last; validation is 2.95%. There are visible historical shocks, notably 19.55% at 2014-07-10 and 4.38% at 2019-10-03.

The relational database has 255,360 users, 333,893 posts, 623,967 comments, 1,175,368 post-history records, 1,317,876 votes, 77,337 links, and 463,463 badge records. Across tables, a calendar day has median 1,099 and 99th-percentile 2,102 events. Vote timestamps are day-granular and strongly tied; post-history also has substantial timestamp ties. Deterministic replay therefore orders by timestamp, table identifier, and primary key.

Null-key behavior is material: 99.6% of votes have null UserId, so vote features and events attach to the voted post and known owner only. OwnerUserId is null on 5,245 posts; UserId is null on 11,679 comments and 75,337 post-history rows. Post links have 16,166 null source posts and 1,749 null related posts. Null graph endpoints are skipped or represented as a self post event without inventing an identity.

## Coverage axes

- Origin and account age: early/small versus late/large snapshots.
- History density: no activity, sparse activity, and high-degree users.
- Recency and trend: 7, 30, 91, and 365-day windows versus lifetime history.
- Interaction role: authoring, answering, commenting, revising, receiving votes/responses, and linking.
- Badge state: no prior badge, recent badge, badge class, and family diversity.
- Graph state: counterpart diversity, thread topology, recent neighbors, and event ordering.
- Data quality: null endpoints, tied timestamps, unseen or graph-isolated users.

## Coverage check

The structural-signal claim remains unverified and is tested through purged forward-fold OOF predictions. PyTorch Geometric 2.8.0 is installed; the implementation retains a plain PyTorch replay path. A100 bf16 GRU throughput measured 1.65 million node updates per second on synthetic 8,192-node batches; the end-to-end pretraining target is conservatively 40,000 graph events per second after attention, loading, and neighbor-store work.

## Reporting strata

The candidate writes count and ROC AUC for activity-density, prior-badge, account-age, and recency strata to `metrics.json`. Bootstrap standard error and prediction-rank correlations are computed after the first full validation run without using them for model selection.
