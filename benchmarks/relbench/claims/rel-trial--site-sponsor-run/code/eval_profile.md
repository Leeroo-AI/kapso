# Evaluation profile

## Mechanics

- The immutable grader runs `main.py --debug` for fast fidelity and `main.py` for full fidelity, then scores all 37,003 validation rows with the official RelBench task evaluator.
- `fraction` and `seed` do not subsample scoring. The score of record is macro `link_prediction_map` at 10; precision and recall at 10 are also reported.
- Candidate code owns all replay, retrieval, ranking, and fallback behavior. The harness controls only the run mode, timeout, contract validation, official scoring, and full-run archival.
- Validation predictions must come from Model A fit without validation labels. Model B may consume validation labels only for test predictions.

## Input distribution

| Split | Rows | Query timestamps | Facilities | Mean labels | Median labels | Maximum labels |
|---|---:|---:|---:|---:|---:|---:|
| Train | 669,310 | 20 | 411,277 | 2.219 | 1 | 241 |
| Validation | 37,003 | 1 | 37,003 | 2.165 | 1 | 171 |
| Test | 27,428 | 1 | 27,428 | unavailable | unavailable | unavailable |

- Train query groups span 2000 through 2019; validation is 2020-01-01 and test is 2021-01-01.
- Validation label-list strata are 22,603 singleton rows, 10,076 rows with 2–3 labels, 3,601 with 4–10, and 723 with more than 10.
- The deduplicated `(facility_id, sponsor_id, nct_id)` chronological join contains 1,927,111 events, 431,513 facilities, and 46,134 sponsors. The raw join has approximately 2.69 million rows.
- Sponsor activity is highly long-tailed: deduplicated study activity quantiles are 1, 1, 1, 3, 9, 121, and 6,731 at 0%, 25%, 50%, 75%, 90%, 99%, and 100%.
- Prior measurement found 45.8% validation facilities with history and 54.2% cold; exact-pair recurrence covers 20.9% of positive pairs, while country-level history covers 77.6%.

## Coverage axes

- Query year and forward temporal shift.
- Warm versus cold facility state and facility history depth.
- Singleton versus multi-positive label lists.
- Exact-pair recurrence versus sponsor exploration.
- City, state, country, and missing-geography strata.
- Sponsor popularity, momentum, lead/collaborator role, agency class, condition, intervention, and study-type compatibility.
- Static evidence versus late-arriving outcomes, analyses, withdrawals, and reported-event aggregates.

## Smoke measurement

- Torch 2.10.0+cu126 and PyG 2.8.0.post1 import successfully on the lane-visible A100.
- A 100,000-event replay completed in 2.439 seconds (41,002 events/s).
- One 256-query, 192-negative backward pass and exact top-512 sponsor retrieval completed; peak allocated GPU memory was 1.35 GB.
- TGN throughput is therefore not the predicted bottleneck. Candidate quality for the majority-cold validation population remains bounded by static/geographic coverage, so deterministic union and cold-aware retrieval are critical.

## Slice reporting

The candidate writes row counts and self-measured internal MAP diagnostics by warm/cold and label-size strata where labels are legally available. Official output remains the complete macro aggregate.
