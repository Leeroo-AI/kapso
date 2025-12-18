# Principle: Graph Edit Distance

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|NetworkX GED|https://networkx.org/documentation/stable/reference/algorithms/similarity.html]]
* [[source::Paper|GED Survey|https://arxiv.org/abs/1904.04053]]
|-
! Domains
| [[domain::Graph_Theory]], [[domain::Similarity_Metrics]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for calculating similarity between workflow graphs using Graph Edit Distance with configurable cost functions.

=== Description ===

Graph Edit Distance (GED) measures the minimum cost to transform one graph into another:

1. **Node Operations**: Insert, delete, or substitute nodes
2. **Edge Operations**: Insert, delete, or substitute edges
3. **Cost Functions**: Configurable costs for each operation type
4. **Optimal Matching**: Finds minimum-cost edit sequence

GED provides:
- **Quantitative Similarity**: Single numeric score for comparison
- **Edit Path**: Specific operations needed for transformation
- **Configurable Strictness**: Different costs for different change types
- **Semantic Awareness**: Type-aware substitution costs

=== Usage ===

Apply this principle when:
- Comparing structured data with insertions/deletions
- Implementing similarity metrics for graphs
- Building evaluation systems for generative models
- Creating diff tools for complex structures

== Theoretical Basis ==

GED is formalized as:

<math>
GED(G_1, G_2) = \min_{\pi \in \Pi(G_1, G_2)} \sum_{e \in \pi} c(e)
</math>

Where:
- <math>\Pi(G_1, G_2)</math> is the set of all valid edit paths
- <math>c(e)</math> is the cost of edit operation <math>e</math>

<syntaxhighlight lang="python">
# Pseudo-code for GED calculation

def calculate_ged(g1, g2, config):
    # 1. Define cost functions
    def node_subst_cost(n1_attrs, n2_attrs):
        if n1_attrs["type"] == n2_attrs["type"]:
            return config.node_substitution_same_type + param_diff_cost
        elif are_similar_types(n1_attrs["type"], n2_attrs["type"]):
            return config.node_substitution_similar_type
        else:
            return config.node_substitution_different_type

    def node_del_cost(n_attrs):
        return config.node_deletion_cost

    def node_ins_cost(n_attrs):
        return config.node_insertion_cost

    # 2. Calculate optimal edit path
    for node_path, edge_path, cost in nx.optimize_edit_paths(
        g1, g2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
    ):
        return node_path, edge_path, cost

    # 3. Convert to similarity: 1 - (cost / max_cost)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
