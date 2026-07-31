Critical path: temporally censored candidate/state artifacts, measured replay rate 41,002 joined events/s; target full 1.93M-event replay under 2 minutes and complete 64K-query retrieval/reranking under 20 minutes.
Confirmation points: contract/debug after data build; 2018 and 2019 forward-fold candidate/score diagnostics before full fit; Model A predictions frozen before Model B continuation.
Freeze time: stop optional objectives by 3h10m elapsed, freeze candidate artifacts by 3h25m, reserve the final 35 minutes for full inference, checks, and registered evaluation.

# Implementation plan

1. Build compact chronological joined events and static hashed geography/text attributes.
2. Train the two-tower exploration head and gated recurrence reranker with forward-only supervision.
3. Produce Model A validation predictions from train labels and Model B test predictions after train+validation continuation.
4. Validate exact output shapes, IDs, distinctness, temporal cutoff assertions, and official metrics.
