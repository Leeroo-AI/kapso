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
- **MISS-NOVEL** — last-product priors — churn statistics of the last product a customer bought, computed only from pre-cutoff rows — improved every validation fold; the customer's own history said less than the censored context of the entity it touched last: [mined/it-1/flow-3.md#judgment]
- **MISS-NOVEL** — recent-product context blocks (aggregates over the products touched in the trailing window, cutoff-safe) added a further measured gain on top of ego features; again the neighbor entity's context, not more of the customer's own volume: [mined/it-1/flow-3.md#judgment]
- **MISS-NOVEL** — co-review propagation — features passed from customers who reviewed the same products, restricted to pre-cutoff events — added a small but positive delta; a third surface of the same move, context imported from a censored neighboring entity: [mined/it-1/flow-2.md#evaluation]

## Claims settlement


## Serving

