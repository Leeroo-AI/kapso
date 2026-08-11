Critical-path artifact + target rate: pinned TabPFN-v2 inference across 55 rolling ticks; target at most 45 seconds per full tick after one checkpoint download.
Planned confirmation points: sidecar debug gate, one static TabPFN timing probe, rolling fast gate, then registered full manifest.
Freeze time: stop model changes after the full-run gate leaves 90 minutes of the 120-minute evaluator timeout for completion and capture.

# Plan

1. Characterize evaluator mechanics, snapshot formats, temporal strata, and dependency availability.
2. Build and validate a compact all-table feature matrix plus logistic and LightGBM sidecars.
3. Pin, hash, and time TabPFN-v2; retain the sidecar fallback if the measured projection exceeds budget.
4. Exercise the rolling contract in fast mode and run the immutable full-fidelity evaluator.
