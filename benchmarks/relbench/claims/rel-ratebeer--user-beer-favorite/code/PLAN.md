TIME ALLOCATION
- Critical path: temporally censored candidate/feature matrices; measured debug rate is 6,400–9,800 episode-candidates/second, revising the achievable target to at least 6,000/second and both legal replay chains by minute 120.
- Confirmation points: deterministic debug artifacts by minute 25, replay/candidate recall and throughput pilot by minute 55, Model A by minute 115, Model B by minute 180.
- Freeze time: minute 205, preserving 20 minutes for contract checks and the foreground full evaluation.

Revision reason: complementary-source construction and 90 pair features made the measured rate lower than the pre-pilot estimate; the observed rate still projects the 17.2M-row chain inside the planned build window.

Second revision: full-candidate folds exposed sampled-negative selection optimism. Cached inference-sized folds now bound score selection, and the freeze moved to minute 190 because cache reuse leaves sufficient evaluation reserve.

# Plan

1. Profile the immutable scorer, seed tables, event-time coverage, episode counts, and legal temporal joins.
2. Implement the deterministic censored retrieval floor and verify full-shape debug outputs.
3. Build cached replay episodes, complementary retrieval, censored features, purged internal folds, and LambdaRank chains A and B.
4. Validate prediction contracts, persist diagnostics and campaign notes, then run the registered full evaluation in the foreground.
