# Implementation: calculate_graph_edit_distance

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|NetworkX GED|https://networkx.org/documentation/stable/reference/algorithms/similarity.html]]
|-
! Domains
| [[domain::Graph_Theory]], [[domain::Similarity_Metrics]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for calculating workflow similarity using Graph Edit Distance with custom cost functions.

=== Description ===

`calculate_graph_edit_distance()` is the main entry point for workflow comparison:

1. **Graph Relabeling**: Both graphs relabeled for structure-based matching
2. **Cost Function Setup**: Creates closures with configuration-based costs
3. **GED Calculation**: Uses NetworkX's `optimize_edit_paths()` for optimal edit sequence
4. **Edit Extraction**: Converts edit path to human-readable operations
5. **Similarity Scoring**: Normalizes cost to 0-1 similarity score

The function handles edge cases like empty graphs and GED calculation failures gracefully.

=== Usage ===

Call this function with two workflow graphs and configuration to get similarity score and detailed edit operations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L19-144

=== Signature ===
<syntaxhighlight lang="python">
def calculate_graph_edit_distance(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    config: WorkflowComparisonConfig
) -> Dict[str, Any]:
    """
    Calculate graph edit distance with custom cost functions.

    Args:
        g1: First workflow graph (generated)
        g2: Second workflow graph (ground truth)
        config: Configuration with cost weights

    Returns:
        Dictionary with:
            - similarity_score: 0-1 (1 = identical)
            - edit_cost: Total cost of edits
            - max_possible_cost: Theoretical maximum cost
            - top_edits: List of most important edit operations
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.similarity import calculate_graph_edit_distance
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| g1 || nx.DiGraph || Yes || Generated workflow graph
|-
| g2 || nx.DiGraph || Yes || Ground truth workflow graph
|-
| config || WorkflowComparisonConfig || Yes || Configuration with cost weights
|}

=== Outputs ===
{| class="wikitable"
|-
! Key !! Type !! Description
|-
| similarity_score || float || 0-1 score (1 = identical)
|-
| edit_cost || float || Total cost of all edit operations
|-
| max_possible_cost || float || Maximum possible cost (delete g1 + insert g2)
|-
| top_edits || List[Dict] || List of edit operations sorted by cost
|}

=== Edit Operation Structure ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| type || str || "node_insert", "node_delete", "node_substitute", "edge_insert", etc.
|-
| description || str || Human-readable description of the edit
|-
| cost || float || Cost of this operation
|-
| priority || str || "critical", "major", or "minor"
|-
| node_name || str || (optional) Name of affected node
|-
| parameter_diff || Dict || (optional) Parameter differences for substitutions
|}

== Usage Examples ==

=== Basic Comparison ===
<syntaxhighlight lang="python">
from src.graph_builder import build_workflow_graph
from src.similarity import calculate_graph_edit_distance
from src.config_loader import load_config

# Load and build graphs
config = load_config()
g1 = build_workflow_graph(generated_workflow, config)
g2 = build_workflow_graph(ground_truth_workflow, config)

# Calculate similarity
result = calculate_graph_edit_distance(g1, g2, config)

print(f"Similarity: {result['similarity_score']:.1%}")
print(f"Edit cost: {result['edit_cost']:.1f}")

for edit in result['top_edits'][:5]:
    print(f"  [{edit['priority']}] {edit['description']} (cost: {edit['cost']:.1f})")
</syntaxhighlight>

=== Similarity Formula ===
<syntaxhighlight lang="python">
# Similarity is calculated as:
similarity_score = 1.0 - (edit_cost / max_possible_cost)

# Where max_possible_cost = cost to delete all of g1 + insert all of g2
max_cost = (
    len(g1.nodes()) * config.node_deletion_cost
    + len(g1.edges()) * config.edge_deletion_cost
    + len(g2.nodes()) * config.node_insertion_cost
    + len(g2.edges()) * config.edge_insertion_cost
)

# Score is clamped to [0.0, 1.0]
</syntaxhighlight>

=== Cost Function Behavior ===
<syntaxhighlight lang="python">
# Node substitution costs vary by type match
def node_substitution_cost(n1_attrs, n2_attrs):
    type1 = n1_attrs.get("type")
    type2 = n2_attrs.get("type")

    if type1 == type2:
        # Same type: low cost + parameter differences
        return config.node_substitution_same_type + param_cost

    elif config.are_node_types_similar(type1, type2):
        # Similar types (in same similarity group)
        return config.node_substitution_similar_type

    else:
        # Different types: high cost
        return config.node_substitution_different_type
</syntaxhighlight>

=== Empty Graph Handling ===
<syntaxhighlight lang="python">
# Both empty: perfect match
if g1.number_of_nodes() == 0 and g2.number_of_nodes() == 0:
    return {
        "similarity_score": 1.0,
        "edit_cost": 0.0,
        "max_possible_cost": 0.0,
        "top_edits": [],
    }
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Graph_Edit_Distance]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
