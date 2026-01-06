# Principle: Similarity Scoring

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Similarity_Metrics]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for normalizing edit costs into a 0-1 similarity score by calculating the theoretical maximum cost as a baseline.

=== Description ===

Similarity Scoring converts raw edit costs to interpretable scores:

1. **Maximum Cost Calculation**: Computes worst-case transformation cost
2. **Normalization Formula**: `similarity = 1 - (edit_cost / max_cost)`
3. **Boundary Handling**: Clamped to [0.0, 1.0] range
4. **Edge Cases**: Empty graphs handled specially

The maximum cost represents complete replacement:
- Delete all nodes from graph 1
- Delete all edges from graph 1
- Insert all nodes from graph 2
- Insert all edges from graph 2

This provides:
- **Intuitive Scale**: 1.0 = identical, 0.0 = completely different
- **Size Independence**: Large workflows don't inherently score lower
- **Configurable Sensitivity**: Different cost weights affect scoring

=== Usage ===

Apply this principle when:
- Converting distance metrics to similarity scores
- Normalizing costs across different-sized inputs
- Creating threshold-based pass/fail metrics
- Building interpretable evaluation scores

== Theoretical Basis ==

The normalization formula:

<syntaxhighlight lang="python">
# Maximum cost = delete g1 entirely + insert g2 entirely

def calculate_max_cost(g1, g2, config):
    # Cost to remove everything from g1
    delete_cost = (
        len(g1.nodes()) * config.node_deletion_cost
        + len(g1.edges()) * config.edge_deletion_cost
    )

    # Cost to add everything in g2
    insert_cost = (
        len(g2.nodes()) * config.node_insertion_cost
        + len(g2.edges()) * config.edge_insertion_cost
    )

    return delete_cost + insert_cost

# Similarity formula
max_cost = calculate_max_cost(g1, g2, config)
if max_cost == 0:
    similarity = 1.0 if edit_cost == 0 else 0.0
else:
    similarity = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
</syntaxhighlight>

Example with default costs:
- g1: 5 nodes, 4 edges
- g2: 6 nodes, 5 edges
- max_cost = (5×10 + 4×5) + (6×10 + 5×5) = 70 + 85 = 155

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_calculate_max_cost]]
