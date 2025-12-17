{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|String Similarity Metrics|https://en.wikipedia.org/wiki/String_metric]]
* [[source::Paper|Normalized Edit Distance|https://doi.org/10.1145/321879.321880]]
|-
! Domains
| [[domain::Similarity_Metrics]], [[domain::Workflow_Analysis]], [[domain::Normalization]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Similarity Calculation is the principle of converting raw edit distance costs into normalized similarity scores between 0 and 1, where 1 indicates identical workflows and 0 indicates completely dissimilar workflows.

=== Description ===

The similarity calculation principle transforms the absolute edit cost from GED computation into an intuitive, normalized metric that users can easily interpret. Raw edit costs are unbounded and context-dependent (larger workflows naturally have higher edit distances), making them difficult to interpret without normalization.

The normalization process addresses several challenges:

1. **Scale Independence**: Workflows of different sizes should produce comparable similarity scores

2. **Bounded Output**: Scores must fall within [0, 1] for intuitive interpretation

3. **Monotonicity**: Lower edit costs should always yield higher similarity scores

4. **Meaningful Bounds**:
   - Similarity = 1.0: Workflows are identical (edit cost = 0)
   - Similarity = 0.0: Workflows are maximally different (edit cost ≥ max_cost)

The formula uses the maximum possible edit cost as a normalizing factor, ensuring that the score reflects the proportion of differences relative to the total workflow size.

=== Usage ===

Apply this principle when:
* Displaying similarity metrics to end users
* Ranking workflows by similarity for search/recommendation
* Setting similarity thresholds for deduplication
* Creating similarity matrices for workflow clustering
* Comparing workflows of different sizes fairly

== Theoretical Basis ==

=== Normalization Formula ===

The similarity score is computed as:

```
similarity_score = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
```

Where:
* `edit_cost`: Sum of costs from GED calculation
* `max_cost`: Maximum possible edit distance between the workflows
* `max()` and `min()`: Ensure bounds of [0, 1]

=== Maximum Cost Calculation ===

The maximum cost represents complete dissimilarity:

```
max_cost = cost_delete_all(G1) + cost_insert_all(G2)

where:
  cost_delete_all(G1) = Σ node_delete_cost + Σ edge_delete_cost
                         v∈V₁              e∈E₁

  cost_insert_all(G2) = Σ node_insert_cost + Σ edge_insert_cost
                         v∈V₂              e∈E₂
```

This represents deleting everything from workflow 1 and inserting everything from workflow 2.

=== Properties ===

1. **Symmetry**: similarity(W₁, W₂) = similarity(W₂, W₁)

2. **Identity**: similarity(W, W) = 1.0

3. **Non-negativity**: similarity(W₁, W₂) ≥ 0

4. **Normalization**: 0 ≤ similarity(W₁, W₂) ≤ 1

5. **Monotonicity**: If edit_cost₁ < edit_cost₂, then similarity₁ > similarity₂

=== Edge Cases ===

```python
# Empty workflows
if |V₁| = 0 and |V₂| = 0:
    similarity = 1.0  # Both empty = identical

# One empty workflow
if |V₁| = 0 or |V₂| = 0:
    similarity = 0.0  # One empty = maximally different

# Zero max_cost (both empty after filtering ignored fields)
if max_cost = 0:
    similarity = 1.0  # No comparable differences
```

=== Interpretation Guide ===

Similarity score interpretation:

```
0.95 - 1.00: Nearly identical (minor parameter differences)
0.85 - 0.95: Very similar (same structure, some parameter changes)
0.70 - 0.85: Similar (same workflow type, notable differences)
0.50 - 0.70: Somewhat similar (shared patterns, different implementation)
0.30 - 0.50: Weakly similar (few common elements)
0.00 - 0.30: Dissimilar (fundamentally different workflows)
```

=== Alternative Metrics ===

Other similarity formulations:

1. **Normalized Edit Distance**:
   ```
   similarity = 1 - (edit_cost / max(|V₁|, |V₂|))
   ```

2. **Jaccard Similarity** (set-based):
   ```
   similarity = |nodes₁ ∩ nodes₂| / |nodes₁ ∪ nodes₂|
   ```

3. **Cosine Similarity** (vector-based):
   ```
   similarity = (v₁ · v₂) / (||v₁|| × ||v₂||)
   ```

The chosen formula balances interpretability with mathematical properties.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_similarity_formula]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_GED_Calculation]]
* [[related::Principle:n8n-io_n8n_Result_Formatting]]
