TIME ALLOCATION: Critical-path artifact: cutoff-safe candidate recall, targeting one complete 12.64M-review scan and directional/popularity indexes within 35 minutes.
TIME ALLOCATION: Confirm at 20 minutes with train-only profile, at 90 minutes with debug contract, and at 180 minutes with end-to-end full-build timing.
TIME ALLOCATION: Freeze implementation by minute 275, reserving the remaining time for the registered foreground evaluation and fixes.

# Plan

1. Characterize the immutable scorer and train-only forward-fold input strata; persist measurements without modifying the protected evaluation suite.
2. Implement cutoff-parameterized histories, directional transitions, affinity/popularity sources, optional ALS/text gates, feature generation, and deterministic ranking.
3. Validate debug artifacts, benchmark the full path, run the registered full evaluation, and capture the manifest.

