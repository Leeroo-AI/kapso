# Evaluation profile

The immutable grader runs `main.py` in an isolated process, requires integer arrays of shape `(1043, 10)` and `(499, 10)`, and computes all official recommendation metrics over the complete validation split. The score of record is row-mean `link_prediction_map`; `fraction` does not subsample this task. Validation labels are reporting-only and may not influence candidate, feature, or parameter selection.

## Input distribution

- Validation: 1,043 rows, one timestamp (`2018-09-01`), 1,043 distinct users, label length mean 5.33, median 2, range 1–397.
- Test: 499 rows, one timestamp (`2020-01-01`), 499 distinct users, labels unavailable.
- Train: 1,099 rows over two origins (`2018-03-05`, `2018-06-03`), 1,032 distinct users, label length mean 3.94, median 1, range 1–143.
- Destination universe: 751,524 integer beer identifiers; ten distinct ranked predictions are required per row.

## Coverage axes

- User history depth: no prior favorites versus sparse or deep favorite history; recent versus dormant rating activity.
- Candidate provenance: global favorite/rating popularity, own highly rated beers, rating co-occurrence, co-favorite users, style/brewer affinity, and place/geography evidence.
- Temporal regime: early sparse favorite supervision for Model A versus the longer 2018–2019 Model B history.
- Label cardinality: singleton, 2–4, 5–10, and more than 10 future favorites.
- Item regime: globally popular versus niche, new versus established, seasonal versus non-seasonal, and cold versus historically rated.

## Mechanics and discrepancies

`availability.created_at` spans 2024–2025 in the sanitized database, after every query origin and the test cutoff. The required `created_at <= origin` censor therefore makes availability-derived inference features empty; the source is audited but blocked rather than leaked. UPC records have no usable event time and are excluded. Mutable entity snapshot aggregates are excluded from modeling.

## Reporting strata

The candidate records candidate-source coverage, history-depth strata, and internal purged-fold MAP in `metrics.json`. Official validation per-stratum scores are not used for selection.
