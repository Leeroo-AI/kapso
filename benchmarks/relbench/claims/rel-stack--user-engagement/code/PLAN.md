Critical-path artifact + target rate: cutoff-keyed MiniLM node encodings and heterogeneous graph materialization; measured 3,433 short comments/s, budget at least 1,500 mixed texts/s and 25 minutes per GNN epoch.
Planned confirmation points: prediction-contract smoke test; 100k-text cache checkpoint; first full graph epoch timing; internal 2020-04/2020-07 AUC and blend stability; final artifact validation.
Freeze time: 2026-08-10 00:55 UTC, preserving the final 15 minutes for the required foreground full evaluator and reporting.

# Plan

1. Characterize evaluation mechanics, input strata, temporal boundaries, installed APIs, and cache state.
2. Implement safe cutoff-specific graph construction, MiniLM encoding, temporal two-hop GraphSAGE, compact causal LightGBM, and independent Model-A/Model-B chains.
3. Use 2020-04 and 2020-07 train-only origins for epoch and blend checks, then refit and persist predictions.
4. Exercise debug mode, verify temporal sampler assertions and file contracts, run full evaluation, and record bootstrap/slice diagnostics.
