TIME ALLOCATION: Critical-path artifact is five-fold TabPFN-v2 OOF predictions; target at least 0.125 completed folds/minute, revised after the 2,000-row CUDA timing.
TIME ALLOCATION: Confirm after the 200-row API smoke test, the 2,000-row timing, the five-fold gate, and one debug insurance run.
TIME ALLOCATION: Freeze implementation by 21:55 UTC, reserving at least 25 minutes for one registered full evaluation and artifact checks.

# Plan

1. Preserve run_0001 byte-for-byte as the fallback and reuse its exact cached causal feature matrix.
2. Characterize the immutable score, five history-volume bands, forward origins, and all-table coverage without inspecting hidden test labels.
3. Smoke-test TabPFN-v2 median and quantile CUDA inference, then time its bounded 2,000-row context.
4. Reconstruct the frozen run_0001 champion across five common forward folds and generate raw/logit TabPFN predictive medians.
5. Select band weights only by leave-one-origin-out meta-cross-fitting plus a 5,000-replicate day-block bootstrap gate.
6. Fit model A through May 4 and model B through May 10, preserving banked predictions for every rejected band or failure.
7. Run one debug gate, validate both vectors, and execute one immutable full-fidelity evaluation.
