# Evaluation profile

## Mechanics

- The immutable harness executes `python main.py` for full fidelity with a 7,200-second cap.
- `fraction` and `seed` are manifest metadata; the scorer always evaluates all 536 validation rows.
- The primary score is validation ROC-AUC. Average precision, accuracy, and F1 are also computed. Test metrics remain hidden.
- Validation predictions must come from a train-only label chain. Test predictions may come from a separately refit train-plus-validation chain.
- Required outputs are finite float vectors in `[0,1]` with shapes `(536,)` and `(420,)`, aligned to task-table order.

## Measured input profile

| Axis | Train | Validation | Test |
|---|---:|---:|---:|
| Rows | 14,442 | 536 | 420 |
| Positives | 505 | 5 | hidden |
| Positive rate | 3.50% | 0.93% | hidden |
| Distinct users | 1,969 | 142 | 64 |
| Distinct events | 8,333 | 464 | 401 |
| Invited rate | 4.35% | 2.99% | 5.95% |
| User/timestamp batches | 4,410 | 234 | 98 |
| Null candidate event ID | 638 | 205 | 0 |

- Seed timestamps cover 2012-04-27 through 2012-11-20 for train, 2012-11-21 through 2012-11-28 for validation, and 2012-11-29 through 2012-12-12 for test.
- The entity table has 14,978 rows at the test cutoff and 15,398 rows in the full autocomplete snapshot. Cutoff primary keys map train/validation; full-snapshot primary keys map test. Timestamp equality is asserted for all mapped rows.
- Friendship has 30,386,403 rows and 217,555 resolved identity edges. Full outgoing list size is available even when friend identity is unresolved.
- Candidate event content has 101 `c_` columns across 8,846 distinct candidate events, with 128,248 nonzero values.
- Only five validation positives make the headline ROC-AUC highly variable; model and blend selection therefore use fixed expanding train-only folds.
- Among rows with a resolved event ID, past-start events cover 7.5% of train, 13.3% of validation, and 4.0% of test. Earlier shared notes reporting 39%/92% used historical foreign keys against the wrong full-snapshot entity mapping and were corrected after cutoff/full join assertions.

## Coverage axes

- Resolved friendship neighbor versus no resolved neighbor.
- Cold user versus user with prior labeled exposure.
- Previously seen exact event versus event cold start.
- Event already started versus future event.
- Invited versus organic response.
- Social diffusion, content-conditioned diffusion, attendance propagation, co-response, demographic, content, and direct temporal feature families.

## Critical path

The bounded artifact is the graph feature bundle rather than head fitting. The measured database load is 3.3 seconds; the target is a cached static representation plus chronological features for all 15,398 seeds in under 15 minutes. Static-cache completion, temporal assertions, and pooled OOF graph lift are the confirmation points.

## Confirmation

- The cached full graph build processes all seeds in under one minute; the one-time DeepWalk cache build completed before full fitting.
- Row-quantile expanding folds contain 7,219/9,025/10,827/12,633 prefix-training rows and 83/67/43/64 positives in their respective forward validation blocks.
- Full graph features scored 0.94359 pooled OOF ROC-AUC, while the scalar topology specialist remained best after family-specific early stopping. Its forward-fold AUCs are 0.96835/0.98626/0.98443/0.97333, better than direct on every block, with a selected median of 68 trees. High-dimensional embeddings and compact supervised propagation were unstable, but direction-specific degree, unresolved list size, reach, PageRank, clustering, and component/community-size features transfer consistently; train-only selection therefore keeps the topology specialist.
