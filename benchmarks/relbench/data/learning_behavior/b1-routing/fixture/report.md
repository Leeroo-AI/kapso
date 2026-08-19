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
- **MISS-UNCARDED** — the lane combined its two model families in rank space — rank_blend = (1-w)*rank(gbdt) + w*rank(gnn), the normalized rank-average preserving [0,1] — and the gain came from blending decorrelated families, not from any single model's features. Shares rank/normalize vocabulary with within-group feature ranking, but the mechanism is ensemble combination in rank space, not feature representation: [mined/it-1/flow-3.md#implementation]
- **MISS-UNCARDED** — a target-history debug attempt failed with KeyError timestamp_ns because pending snapshots omitted their origin timestamp, and adding that column fixed the retry; seen once in this campaign — a snapshot-schema hygiene phenomenon awaiting recurrence: [mined/it-1/flow-0.md#difficulties]
- **MISS-UNCARDED** — LightGBM warned that negative categorical values were converted to NaN; these were intentional -1 missing-category sentinels and all fits completed correctly — a no-action tool warning that changed nothing: [mined/it-1/flow-0.md#difficulties]

## Claims settlement


## Serving

