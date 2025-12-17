{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Graph_Isomorphism]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for relabeling graph nodes using structural identifiers instead of names, provided by the n8n workflow comparison system.

=== Description ===

The `_relabel_graph_by_structure` function transforms workflow graphs by replacing node names with structure-based identifiers (e.g., "trigger_0", "node_1"). This normalization is critical for accurate graph comparison because:
* It removes bias from node naming differences
* It enables structure-focused comparison rather than name-focused
* It separates trigger nodes from regular nodes for proper alignment
* It maintains a reverse mapping for translating results back to original names

=== Usage ===

Use this implementation when you need to:
* Normalize graphs before computing edit distance
* Compare workflows with different node naming conventions
* Ensure graph comparison focuses on structure, not labels
* Maintain bidirectional mapping between structural and original names

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L421-501

=== Signature ===
<syntaxhighlight lang="python">
def _relabel_graph_by_structure(graph: nx.DiGraph) -> tuple[nx.DiGraph, Dict[str, str]]:
    """Relabel graph nodes using structural IDs instead of names.

    Args:
        graph: NetworkX directed graph with original node names

    Returns:
        Tuple of:
        - Relabeled graph with structural IDs (trigger_0, node_1, etc.)
        - Reverse mapping dictionary {structural_id: original_name}
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.similarity import _relabel_graph_by_structure
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| graph || nx.DiGraph || Yes || NetworkX directed graph with original node names and 'is_trigger' node attribute
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| relabeled_graph || nx.DiGraph || Copy of the graph with nodes relabeled using structural IDs
|-
| reverse_mapping || Dict[str, str] || Dictionary mapping structural IDs back to original node names
|}

=== Node Labeling Scheme ===
{| class="wikitable"
|-
! Node Type !! Label Pattern !! Example
|-
| Trigger nodes || trigger_{index} || trigger_0, trigger_1
|-
| Regular nodes || node_{index} || node_0, node_1, node_2
|}

== Usage Examples ==

=== Basic Graph Relabeling ===
<syntaxhighlight lang="python">
from src.graph_builder import build_workflow_graph
from src.compare_workflows import load_workflow
from src.similarity import _relabel_graph_by_structure

# Build graph with original node names
workflow = load_workflow("workflows/my_workflow.json")
graph = build_workflow_graph(workflow)

print("Original nodes:", list(graph.nodes()))
# Output: ['Webhook', 'HTTP Request', 'Set Variable', 'Respond']

# Relabel by structure
relabeled, reverse_map = _relabel_graph_by_structure(graph)

print("Relabeled nodes:", list(relabeled.nodes()))
# Output: ['trigger_0', 'node_0', 'node_1', 'node_2']

print("Reverse mapping:", reverse_map)
# Output: {'trigger_0': 'Webhook', 'node_0': 'HTTP Request', ...}
</syntaxhighlight>

=== Using Relabeling for Comparison ===
<syntaxhighlight lang="python">
import networkx as nx
from src.similarity import _relabel_graph_by_structure

# Build two graphs with different node names
graph1 = build_workflow_graph(workflow1)
graph2 = build_workflow_graph(workflow2)

# Relabel both graphs
g1_relabeled, g1_map = _relabel_graph_by_structure(graph1)
g2_relabeled, g2_map = _relabel_graph_by_structure(graph2)

# Now comparison is structure-based, not name-based
edit_distance = nx.graph_edit_distance(g1_relabeled, g2_relabeled)

print(f"Structural edit distance: {edit_distance}")
print(f"Graph 1 mapping: {g1_map}")
print(f"Graph 2 mapping: {g2_map}")
</syntaxhighlight>

=== Translating Edit Operations Back to Original Names ===
<syntaxhighlight lang="python">
from src.similarity import _relabel_graph_by_structure

# Relabel graphs
g1_relabeled, g1_map = _relabel_graph_by_structure(graph1)
g2_relabeled, g2_map = _relabel_graph_by_structure(graph2)

# Perform comparison (returns operations with structural IDs)
edit_path = nx.optimize_edit_paths(g1_relabeled, g2_relabeled)

# Translate structural IDs back to original names
def translate_operation(structural_id, mapping):
    """Convert structural ID back to original name."""
    return mapping.get(structural_id, structural_id)

# Example: translating a node substitution operation
structural_op = ("node_0", "node_1")
original_op = (
    translate_operation(structural_op[0], g1_map),
    translate_operation(structural_op[1], g2_map)
)

print(f"Structural: {structural_op}")
print(f"Original: {original_op}")
</syntaxhighlight>

=== Analyzing Relabeling Results ===
<syntaxhighlight lang="python">
from src.similarity import _relabel_graph_by_structure

# Build and relabel graph
graph = build_workflow_graph(workflow)
relabeled, reverse_map = _relabel_graph_by_structure(graph)

# Analyze the relabeling
trigger_count = sum(1 for node in relabeled.nodes() if node.startswith('trigger_'))
regular_count = sum(1 for node in relabeled.nodes() if node.startswith('node_'))

print(f"Total nodes: {relabeled.number_of_nodes()}")
print(f"Trigger nodes: {trigger_count}")
print(f"Regular nodes: {regular_count}")

# Show mapping
print("\nStructural ID -> Original Name:")
for struct_id, orig_name in sorted(reverse_map.items()):
    node_data = relabeled.nodes[struct_id]
    print(f"  {struct_id:12s} -> {orig_name:30s} (type: {node_data.get('type', 'N/A')})")
</syntaxhighlight>

=== Verifying Graph Preservation ===
<syntaxhighlight lang="python">
import networkx as nx

# Original graph
graph = build_workflow_graph(workflow)
original_edges = set(graph.edges())

# Relabeled graph
relabeled, reverse_map = _relabel_graph_by_structure(graph)

# Create forward mapping
forward_map = {v: k for k, v in reverse_map.items()}

# Verify edge structure is preserved
relabeled_edges_translated = {
    (reverse_map[u], reverse_map[v]) for u, v in relabeled.edges()
}

assert original_edges == relabeled_edges_translated, "Edge structure not preserved!"
print("Graph structure verified: relabeling preserves all edges")

# Verify node attributes are preserved
for orig_node in graph.nodes():
    struct_node = forward_map[orig_node]
    assert graph.nodes[orig_node]['type'] == relabeled.nodes[struct_node]['type']
    assert graph.nodes[orig_node]['is_trigger'] == relabeled.nodes[struct_node]['is_trigger']

print("Node attributes verified: all attributes preserved")
</syntaxhighlight>

== Implementation Details ==

=== Separation of Triggers and Regular Nodes ===
The function separates nodes into two categories based on the `is_trigger` attribute:
1. '''Trigger nodes:''' Labeled as trigger_0, trigger_1, ...
2. '''Regular nodes:''' Labeled as node_0, node_1, ...

This separation ensures that triggers are aligned with triggers and regular nodes with regular nodes during comparison.

=== Ordering Stability ===
The relabeling maintains consistent ordering:
* Triggers are processed first, in graph iteration order
* Regular nodes are processed second, in graph iteration order
* This creates deterministic structural IDs for reproducible comparisons

=== Attribute Preservation ===
The relabeling process preserves all node and edge attributes from the original graph, including:
* Node type
* is_trigger flag
* Position information
* Parameters
* Connection types

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Graph_Relabeling]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Related Concepts ===
* [[related::Concept:Graph_Isomorphism]]
* [[related::Concept:Graph_Edit_Distance]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]
* [[requires_dependency::networkx]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Graph_Algorithms]]
