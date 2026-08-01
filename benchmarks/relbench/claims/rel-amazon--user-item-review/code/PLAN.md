Critical path: cutoff-safe candidate pool and feature rows, target sustained generation above 2,000 seed rows/minute.
Confirmation points: 2,000-row debug; three training-origin recall@200/800; frozen Model A; full contract check.
Freeze time: stop model/design changes by T+255 minutes and reserve the final 60 minutes for full inference, validation, and evaluation.

# Plan

- Profile the immutable metric mechanics and training-only input strata.
- Build projected events, metadata indices, fixed-source candidates, and replay diagnostics.
- Train the non-text LambdaRank pipeline, then add BGE only if replay-stable.
- Freeze Model A validation predictions, refit Model B with validation labels, and write test predictions.
- Run contract checks and the registered full evaluation.
