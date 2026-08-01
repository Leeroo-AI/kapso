# Evaluation profile

The immutable evaluation suite under `kapso_evaluation/` is read-and-execute only, so this profile is kept at repository root to avoid altering protected evaluation files.

## Mechanics

- Both fast and full fidelity score all 257,939 validation seed rows; `fraction` and `seed` are manifest metadata only.
- The official RelBench task computes macro link-prediction precision, recall, and MAP at rank 10. The primary score is untransformed validation MAP.
- Recommendation artifacts must be integer arrays of shapes `(257939, 10)` and `(292609, 10)` with original task row alignment.
- The scorer does not control retrieval, model inference, ranking ties, or backfill. Full mode invokes `python main.py` with an eight-hour child timeout.

## Input coverage axes

- Time/regime: quarterly training origins from 2008 through 2015, Q4-2015 validation, and Q1-2016 test.
- User history: cold versus warm, activity count, recency, and preference concentration.
- Item history: head versus tail, recent velocity, seasonality, and first/last-review age.
- Relationship: directional successors, brand-group/category affinity, prior interaction, and source overlap.
- Text/product: category path serialization, brand-group strings, titles, descriptions, summaries, and review text.
- Labels: list length and whether positives are reachable by each candidate source at 100, 300, and 600 candidates.

Measured train-only distribution and forward-fold statistics are appended after profiling.

## Measured profile

- The training table has 3,667,157 rows at 31 quarterly origins from 2008-01-10 through 2015-07-02. Origin size grows from 20,684 to 299,025 seeds; mean label size grows from 1.67 to 2.01 while the median remains one.
- Reviews span 2008-01-01 through the censored 2016-01-01 boundary: 12,644,508 rows, 1,584,084 observed customers, 416,125 observed products, 7,719,826 five-star rows, mean rating 4.344, and 70.43% verified.
- At the 2015-07-02 train-only fold, 256,724 of 299,025 seeds are warm (85.85%). Only 1,492 of 601,909 positives repeat a previously reviewed product (0.248%). At 2015-01-01, 80.60% are warm and 0.219% of positives repeat.
- Recent 91-day five-star popularity at 2015-07-02 has recall 7.31%, 10.46%, and 13.82% at widths 100, 300, and 600. Its MAP@10 is 0.01134 on this train-only fold.
- Directional transitions using the last 15 source items and top 30 successors have recall 10.03%, 11.46%, and 11.53% at widths 100, 300, and 600 on 2015-07-02. Brand-group/leaf-category affinity has 6.89%, 7.49%, and 7.49% recall.
- Product categories are arrays: 477,116 products have depth three, with 670 distinct leaf values and 929 complete paths. There are 239,719 distinct non-null `brand` strings; samples mix author-page labels, publishers, and generic values, so the field is treated as a brand group rather than an author identity.
- Product metadata missingness is 3.73% for category, 0.40% for brand group, and 36.53% for description. Price is complete with 10th/50th/90th percentiles 5.99/12.99/25.50.

The warm-user and repeat assumptions agree with the supplied claims. The claimed 0.00841 popularity MAP and 26.8% heuristic-union recall are not universal constants: the cutoff-matched latest training fold produced 0.01134 MAP for 91-day popularity and source-specific recall below the claimed union, so final retention is determined by the two train-only fold gates.

## Forward-fold source strata

| Fold | Source | Recall@100 | Recall@300 | Recall@600 |
|---|---|---:|---:|---:|
| 2015-01-01 | directional | 0.08846 | 0.09783 | 0.09783 |
| 2015-01-01 | ALS | 0.04213 | 0.05029 | 0.05029 |
| 2015-01-01 | semantic | 0.02954 | 0.03015 | 0.03015 |
| 2015-01-01 | affinity | 0.08082 | 0.08082 | 0.08082 |
| 2015-01-01 | popularity | 0.07140 | 0.11069 | 0.14206 |
| 2015-07-02 | directional | 0.10088 | 0.11158 | 0.11158 |
| 2015-07-02 | ALS | 0.04784 | 0.05854 | 0.05854 |
| 2015-07-02 | semantic | 0.03491 | 0.03569 | 0.03569 |
| 2015-07-02 | affinity | 0.08973 | 0.08973 | 0.08973 |
| 2015-07-02 | popularity | 0.06892 | 0.10611 | 0.13680 |

Core-union recall was 0.22260 and 0.23085 on the two folds. ALS increased it by 0.00665 and 0.00986; semantic retrieval then added 0.00021 and 0.00075. Both sources therefore failed the predetermined +0.015 mean-gain gate without reference to official validation labels. The train-only selected tree count was 103.

The ranker audit revealed an aggregation mismatch: early stopping on sampled groups with reachable positives reported 0.76501 MAP, but scoring the January-trained ranker over every July seed produced only 0.000517 macro MAP. The predefined directional-score RRF fallback produced 0.01868 and 0.02079 macro MAP on January and July, including cold/warm July strata of 0.00784/0.02292. The fallback is therefore selected entirely from training folds.

Target-aligned widening retained only designs improving both folds: five-star popularity ordering scored 0.01899/0.02124, five-star-only successors scored 0.01961/0.02201, and using all later events within the capped 50-item history scored 0.02024/0.02310 before coefficient calibration. The frozen RRF (`cf_rank + 0.02*log1p(cf_score) + 2*affinity_rank`) scores 0.02161/0.02367; popularity remains the deterministic tie/backfill order without an additive warm-user boost.

Affinity decomposition showed that category retrieval diluted brand-group specificity: brand-only scoring increased the folds to 0.02333/0.02618 at the old weight and 0.02452/0.02760 at the frozen brand weight 4. Expanding from five to ten recent brand groups changed MAP by less than 0.00005 per fold, so the five-group version won the prescribed simplicity tie-break.

Rank-quantile negative sampling reduced LambdaRank's sampled conditional MAP from 0.765 to 0.418 and improved all-seed July macro MAP from 0.00052 to 0.00262, but it remained far below the 0.02760 RRF and was rejected. Popularity-window, brand-frequency, brand-size, transition-mix, same-day symmetry, per-brand width, and within-brand BGE variants all either lost a fold or remained inside the 0.001 tie band.
