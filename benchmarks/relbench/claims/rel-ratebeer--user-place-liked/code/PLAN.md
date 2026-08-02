Critical-path artifact + target rate: candidate recall@600 from sparse PPR/ALS, benchmark at >=100 queries/minute/checkpoint and cover every inference row.
Planned confirmation points: snapshot build/ALS timing, forward-fold recall and warm/cold MAP, debug contract, then full official evaluation.
Freeze time: reserve the final 30 minutes for prediction validation, foreground official scoring, artifact logging, and handoff.

Revision: internal recall lost roughly half its coverage from first-quarter to fourth-quarter queries under annual snapshots, so the critical-path schedule now includes midyear snapshots before additional ranker work.

# Plan

1. Characterize official metric mechanics, split/origin distributions, user/place warmth, and geographic strata.
2. Build censored annual/exact heterogeneous snapshots, two ALS systems, approximate PPR, safeguards, and caches.
3. Build official-precedence episodes and forward-fold LambdaMART selection, then preserve model-A validation predictions and refit model B.
4. Validate debug/full contracts, run the registered full evaluator, and record per-stratum diagnostics.
