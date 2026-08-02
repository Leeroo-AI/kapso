# Evaluation profile

## Mechanics

- The registered full evaluator runs `python main.py` in an isolated subprocess, requires full aligned validation and test arrays, and scores the complete 293,823-row validation table with official RelBench metrics.
- The primary search metric is row-level accuracy. Macro F1, micro F1, and MRR are also reported. Test labels are absent and test metrics remain hidden.
- Validation predictions must come from train-only histories, fitted weights, and gate parameters. Test predictions may use a separate train-plus-validation refit with unchanged hyperparameters.

## Input distribution

- Train has 1,622,787 rows from 2018-01-02 through 2020-01-31; validation has 293,823 rows from 2020-02-01 through 2020-06-30; test has 400,206 rows from 2020-07-01 through 2020-12-31.
- There are 34 classes. The leading class is 64.43% of train and 60.95% of validation.
- The joined input has 34 organizations, 187,536 products, 21 item categories, 13 document types, 3 channels, 31 billing companies, and 27 currencies.
- Organization-only historical mode has complete validation coverage and 0.9981655623 validation accuracy, with 539 errors. Errors are concentrated in organizations 2000, 0700, 0300, and 0010.
- 99.93% of multi-item documents are single-plant. Validation error slicing therefore needs month, organization, document size, single/multi-item, product support, and multi-plant-exception status where labels are permitted.

## Coverage axes

- Time: three internal forward origins, validation month, and recency at 90-day and 180-day half-lives.
- Cardinality/support: seen versus cold organization, product, interaction, party, and geography keys.
- Hierarchy: header organization/channel/type/company/currency; item product/category/position; four party-to-address roles; document size and unique item signatures.
- Reliability: support, purity margin, entropy, posterior age, and disagreement among sibling item factors.
- Output: original task row order, all 34 classes, finite normalized float32 scores for every row.

## Critical path

The score is bounded by the quality and causal correctness of the history maps feeding the rare organization-exception decisions. The measured organization baseline establishes a 539-row loss budget; all factor and gate changes are selected only from the three training origins, with validation used solely for final reporting.
