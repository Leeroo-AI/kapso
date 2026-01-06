# Implementation: _calculate_max_cost

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

Concrete function for calculating the theoretical maximum edit cost used to normalize similarity scores.

=== Description ===

`_calculate_max_cost()` computes the worst-case transformation cost:

1. **Deletion Cost**: Sum of costs to delete all nodes and edges from g1
2. **Insertion Cost**: Sum of costs to insert all nodes and edges from g2
3. **Total**: Sum of deletion and insertion costs

This represents the cost of completely discarding g1 and building g2 from scratch, which is the maximum possible edit distance.

=== Usage ===

This function is called internally by `calculate_graph_edit_distance()` to normalize the edit cost into a similarity score. It's not typically called directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L195-220

=== Signature ===
<syntaxhighlight lang="python">
def _calculate_max_cost(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    config: WorkflowComparisonConfig
) -> float:
    """
    Calculate theoretical maximum edit cost.
    This represents the cost of completely transforming g1 to g2.

    Args:
        g1: First graph
        g2: Second graph
        config: Configuration

    Returns:
        Maximum possible cost
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function - not typically imported directly
from src.similarity import _calculate_max_cost
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| g1 || nx.DiGraph || Yes || First (generated) workflow graph
|-
| g2 || nx.DiGraph || Yes || Second (ground truth) workflow graph
|-
| config || WorkflowComparisonConfig || Yes || Configuration with cost values
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Type !! Description
|-
| float || Maximum possible edit cost
|}

=== Cost Formula ===
{| class="wikitable"
|-
! Component !! Formula !! Default Cost
|-
| Node deletion || len(g1.nodes) × node_deletion_cost || 10.0 per node
|-
| Edge deletion || len(g1.edges) × edge_deletion_cost || 5.0 per edge
|-
| Node insertion || len(g2.nodes) × node_insertion_cost || 10.0 per node
|-
| Edge insertion || len(g2.edges) × edge_insertion_cost || 5.0 per edge
|}

== Usage Examples ==

=== Implementation ===
<syntaxhighlight lang="python">
def _calculate_max_cost(g1, g2, config):
    # Worst case: delete all of g1, insert all of g2
    delete_cost = (
        len(g1.nodes()) * config.node_deletion_cost
        + len(g1.edges()) * config.edge_deletion_cost
    )
    insert_cost = (
        len(g2.nodes()) * config.node_insertion_cost
        + len(g2.edges()) * config.edge_insertion_cost
    )

    return delete_cost + insert_cost
</syntaxhighlight>

=== Example Calculation ===
<syntaxhighlight lang="python">
# Given:
# g1: 3 nodes, 2 edges (generated workflow)
# g2: 4 nodes, 3 edges (ground truth workflow)
# config: default costs (node=10, edge=5)

delete_cost = 3 * 10 + 2 * 5  # 30 + 10 = 40
insert_cost = 4 * 10 + 3 * 5  # 40 + 15 = 55
max_cost = 40 + 55  # 95

# If actual edit_cost is 20:
similarity = 1 - (20 / 95)  # ≈ 0.79 or 79%
</syntaxhighlight>

=== Similarity Score Usage ===
<syntaxhighlight lang="python">
# In calculate_graph_edit_distance()

max_cost = _calculate_max_cost(g1, g2, config)

# Avoid division by zero
if max_cost == 0:
    similarity_score = 1.0 if edit_cost == 0 else 0.0
else:
    # Clamp to [0.0, 1.0]
    similarity_score = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))

return {
    "similarity_score": similarity_score,
    "edit_cost": edit_cost,
    "max_possible_cost": max_cost,
    ...
}
</syntaxhighlight>

=== Impact of Different Configs ===
<syntaxhighlight lang="python">
# Strict config (higher costs)
strict_config = WorkflowComparisonConfig(
    node_insertion_cost=15.0,
    node_deletion_cost=15.0,
    edge_insertion_cost=8.0,
    edge_deletion_cost=8.0,
)
# Same graphs would have higher max_cost
# → Smaller similarity difference for same edit_cost

# Lenient config (lower costs)
lenient_config = WorkflowComparisonConfig(
    node_insertion_cost=5.0,
    node_deletion_cost=5.0,
    edge_insertion_cost=2.0,
    edge_deletion_cost=2.0,
)
# Same graphs would have lower max_cost
# → Larger similarity difference for same edit_cost
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Similarity_Scoring]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
