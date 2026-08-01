Critical-path artifact + target rate: cached all-table origin feature matrices; sustain at least 150,000 seed rows/minute and finish all required origins by minute 95.
Planned confirmation points: feature throughput/core fold at minute 35; three forward-fold block/weight/objective comparison by minute 125; contract/debug validation by minute 155.
Freeze time: minute 205, leaving 15 minutes for the foreground registered evaluation handoff and manifest capture.

# Measurement

- The immutable grader runs `main.py`, scores all 39,015 validation rows through the official task evaluator, archives full runs, and reports validation R² as the score of record.
- Validation and test each contain one origin; training contains eight complete half-year origins. Targets are positive-only and heavy-tailed.
- Coverage axes are feature origin, author history/cold start, publication momentum, team-size regime, cohort-roster overlap, citations, category behavior, paper/author metadata, and global publication regime.

# Build

Revision after the first internal run: four folds now alternate strict 183-day label embargoes and 365-day temporal-shift gaps. The all-365-day pilot underrepresented the final fits, which legitimately include labels ending at their prediction cutoff; the revised schedule retains two explicit 12-month confirmation folds while measuring the ordinary embargo regime twice.

1. Recreate exact positive-only labels for official and quarterly derived origins with label-window eligibility checks.
2. Cache temporally censored all-table feature matrices while preserving `_row_id`.
3. Compare core/full feature blocks and recency weighting on three 12-month-gap forward folds.
4. Gate objective diversity by forward-OOF complementarity and one-standard-error improvement.
5. Fit separate validation and test models, assert provenance and contract, then run the registered evaluator.
