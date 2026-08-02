Critical path: temporally censored 800-item candidate banks; measured rate is 2,245 user-origin sets/minute including 93-feature construction, replacing the untested 20,000/minute target.
Confirmation points: measured full-channel recall@800 was 22.66% on 1,000 groups; recheck after quota rebalance and origin-local collaborative fits, then contract-check before full evaluation.
Freeze time: 190 elapsed campaign minutes for candidates/features, leaving at least 35 minutes for final A/B generation, checks, and registered evaluation; rate target revised because GPU top-k and Python pair assembly dominate.

# Plan

1. Profile evaluator, task origins, histories, labels, and relational coverage.
2. Build the censored state/candidate/feature pipeline and forward-origin diagnostics.
3. Fit Model A and Model B separately, save and validate predictions.
4. Run the immutable registered full evaluation and record its manifest.
