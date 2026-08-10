TIME ALLOCATION 1/3 — Critical path: frozen chronological memory snapshots for 42 origins; measured GRU core at 1.65M node updates/s, budgeted end-to-end target at least 40k graph events/s.
TIME ALLOCATION 2/3 — Confirmation points: debug contract by 21:35 UTC, frozen-memory/compact forward-fold check by 22:45 UTC, full prediction build by 00:15 UTC.
TIME ALLOCATION 3/3 — Feature freeze at 00:15 UTC on 2026-08-10; reserve through 01:15 UTC for full evaluator, diagnostics, and one repair attempt.

# Plan

- [x] Persist measurement profile and graph/data facts.
- [x] Build deterministic typed events and compact temporal features.
- [x] Train and replay the TGN-style memory encoder.
- [x] Fit purged OOF hazard, residual LightGBM, and logistic meta-model.
- [x] Verify debug/full prediction contracts and run registered evaluation.
- [x] Bootstrap validation resolution, report slices, and update living records.

Revision at 21:31 UTC: the first full run left substantial reserve and localized losses to cold/young users, so the freeze moved to 22:05 UTC for one predeclared training-fold-tested expansion. Promotion requires improvement on both purged folds; anniversary/cadence plus graph passed at +0.001420 and +0.001340 AUC.

Revision at 21:52 UTC: post-threshold state was the remaining unmodeled badge mechanism and the expanded matrix built in under two minutes. Its legal fold gains were +0.000934 and +0.000545, so it passed the same two-fold rule; final freeze moved to 22:20 UTC with cached consumers preserving evaluation reserve.

Revision at 22:10 UTC: all-history supervision was measured against recent windows after old-regime dilution remained visible. A 16-origin LightGBM window improved both folds by +0.001715/+0.001156 and was steadier than the essentially tied 12-origin mean; the hazard restriction failed the two-fold rule and was rejected. Final freeze is 22:35 UTC.

Revision at 22:29 UTC: OOF/final capacity mismatch was audited without changing final capacity. Matching OOF to 500 trees added +0.000987/+0.000779 AUC, so cached final trees remain fixed and only meta-model OOF inputs are aligned. Freeze remains 22:35 UTC.

Final freeze at 23:25 UTC: all later feature/ensemble candidates failed the both-fold rule. The v5 contract is immutable for final audit; `run_0010` is its registered evaluation.
