Critical-path artifact + target rate: temporally audited 96-dimensional route embeddings, at least 20,000 supervised seeds/minute after JIT warm-up; measured 52,000-68,000 seeds/second, so the target was confirmed without revision.
Confirmation points: 100-seed temporal audit, 20,000-seed debug throughput, Encoder A completion, Encoder B completion, prediction contract check.
Freeze time: reserve the final 30 minutes for fallback/fusion selection, full registered evaluation, manifest capture, and documentation.

# Plan

1. Materialize compact projected event indexes and immutable structural mappings in the shared cache.
2. Build and audit the two legal temporal encoder chains at their independent cutoffs.
3. Extract hazard and graph representations, select only with training-only forward anchors, and fit the two prediction chains.
4. Run debug and contract checks, then the registered full-fidelity evaluation.
