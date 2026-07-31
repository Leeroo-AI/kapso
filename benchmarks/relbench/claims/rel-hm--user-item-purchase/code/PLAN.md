TIME ALLOCATION: Critical-path artifact is a valid Model A neural retrieval bank at at least 3,000 queries/s preprocessing and 1,000 queries/s training.
TIME ALLOCATION: Confirm after CUDA smoke test, 25k-query preprocessing benchmark, first 100 optimizer steps, and first internal forward-origin retrieval score.
TIME ALLOCATION: Freeze architecture by hour 3.5, bank Model A validation predictions by hour 4.5, and reserve the final hour for Model B, checks, and official evaluation.

# Plan

1. Measure evaluator mechanics and metadata-only distribution strata.
2. Bank cutoff-censored popularity, cohort, repeat, family, and transition channels.
3. Build daily-basket episodes and validate the Transformer forward/backward path.
4. Train Model A on official training episodes and select epoch count by recent train-only forward origins.
5. Retrieve neural and relational candidates, fit the compact train-only reranker, and save Model A validation predictions.
6. Continue Model B for one fixed epoch with train plus validation episodes, refit the reranker, and predict test.
7. Validate both arrays and run the registered full evaluator.
