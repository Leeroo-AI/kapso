Critical path: cutoff-specific candidate recall at 400; target 30,000 seed rows/minute after event-state construction.
Confirmation points: internal origins 2015-04-30 and 2015-05-04 candidate diagnostics, then fixed-round LambdaRank MAP against RRF.
Freeze time: 2026-07-31 02:50 UTC, preserving at least 20 minutes for full decoding, contract checks, and registered evaluation completion.

# Plan

1. Measure the immutable metric, split origins, history strata, and channel ceilings without accessing test labels.
2. Build temporally censored event states and five-channel candidates for all official origins.
3. Bank RRF predictions, compute all-table rank features, and select a fixed tree count on the last training origin.
4. Fit model A on training labels and preserve its validation predictions; fit model B on training plus validation labels for test only.
5. Validate prediction artifacts and run the registered full-fidelity evaluation in the foreground.
