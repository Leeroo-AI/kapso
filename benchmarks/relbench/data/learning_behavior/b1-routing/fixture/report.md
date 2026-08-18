---
trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
bank_head: fixture
brief: brief.md
hindcast:
  foresight: 0.00
  accuracy: null
  serving: null
  score: 0.00
  rationale: >-
    Fixture report; classes engineered per truth.md.
---

## Extraction
- **MISS-UNCARDED** — activity percentile within the customer's signup cohort outperformed the raw activity count wherever both were offered to the model: the label rewards standing relative to the competing cohort, and the absolute count cannot see it. Same mechanism family as within-group ranking of informative features: [mined/it-1/flow-3.md#judgment]
- **MISS-UNCARDED** — grouping rows by customer when building the validation split (group-aware folds) changed the relative standing of candidate models: ungrouped folds leaked a customer across sides and inflated whichever model memorized that group. The mechanism is evaluation integrity under grouped rows, not feature construction: [mined/it-1/flow-2.md#evaluation]
- **MISS-UNCARDED** — one lane observed that pushing ORDER BY into the DuckDB window scan made the ego-feature pass roughly three times faster; seen once, mechanism not isolated (could be spill avoidance or partition pruning): [mined/it-1/flow-0.md#difficulties]
- **MISS-UNCARDED** — the sandbox package mirror returned 502 twice during environment setup and plain retries fixed it; operational noise with no bearing on modeling: [mined/operations.md]

## Claims settlement


## Serving

