# Evaluation profile

## Mechanics

- Registered command: `python kapso_evaluation/kapso_eval.py --fidelity full --fraction 1.0 --seed 1337`.
- The grader runs `python main.py`, loads both NumPy files without pickle, checks finite row-aligned outputs, and calls the official RelBench evaluator on all 39,015 validation rows. The optimization score is validation accuracy; multiclass F1 is also reported. Test labels and metrics are absent from the candidate-visible loop.
- Full fidelity archives predictions and a candidate-code snapshot. The fraction and seed are manifest metadata and do not subsample scoring rows. The child timeout is 14,400 seconds.
- Validation predictions must be produced by a chain fit without validation labels. Test predictions may use validation labels and other outcomes whose 182-day label windows end by the 2023-01-01 cutoff.

## Input distribution

- Train contains 210,769 rows at eight semiannual origins from 2018-01-06 through 2021-07-03. Origin sizes increase from 13,008 to 38,657. Validation is 39,015 rows at 2022-01-01 and test is 39,655 rows at 2023-01-01.
- There are 53 output columns. Observed train targets occupy 52 labels and validation targets occupy 51 labels. The train head is labels 3, 52, 4, 6, and 10; the validation head is 3, 52, 18, 4, and 10, demonstrating temporal prior drift.
- Cold-history rates by train origin are 98.17%, 66.66%, 57.37%, 53.64%, 48.23%, 47.86%, 34.37%, and 30.95%. Validation is 28.14% cold and test is 28.75% cold.
- Among warm validation rows, author-history age has quartiles 376, 767, and 1,194 days. Among warm test rows, the quartiles are 642, 1,061, and 1,529 days.
- Authorship team size is heavy-tailed: median 2, 90th percentile 6, 99th percentile 19, and maximum 2,825. Coauthor graph construction therefore needs bounded fan-out.

## Coverage axes

- Time origin and label-prior regime.
- Cold versus warm authors, plus warm-history age and activity.
- Head versus tail category.
- Own-history strength, recency, persistence, trend, and entropy.
- Relational-channel availability: coauthor, cited-paper, citing-paper, and secondary-category history.
- Identity evidence: name, ORCID, causal Author_ID cohort, and debut timing.
- Within-author trend direction and local causal Author_ID neighborhood priors, especially for the approximately 28% cold cohort.
- Small versus very large author teams.

The supplied 28.1% cold-validation assumption is confirmed. The measured test cold rate is similar, while warm test histories are substantially older than warm validation histories. Candidate training forcibly inserts the true class, so sampled-group positive recall is exactly 100%; inference expands every row to all 53 candidates.
