# Implementation: _relabel_graph_by_structure

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|NetworkX|https://networkx.org/documentation/stable/]]
|-
! Domains
| [[domain::Graph_Theory]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for relabeling graph nodes from display names to structural identifiers for name-independent comparison.

=== Description ===

`_relabel_graph_by_structure()` creates a canonically labeled copy of the graph:

1. **Node Sorting**: Sorts nodes by type, out-degree (descending), in-degree (descending), then name
2. **Trigger Separation**: Identifies trigger nodes via `is_trigger` attribute
3. **Sequential Labeling**: Assigns `trigger_0`, `trigger_1`, ... and `node_0`, `node_1`, ...
4. **Graph Relabeling**: Uses `nx.relabel_nodes()` to create copy with new labels
5. **Attribute Preservation**: Stores `_original_name` and `_name_hash` in node attributes
6. **Quote Normalization**: Handles smart quotes for consistent name hashing

=== Usage ===

This function is called internally by `calculate_graph_edit_distance()` before GED calculation. It's not typically called directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L421-501

=== Signature ===
<syntaxhighlight lang="python">
def _relabel_graph_by_structure(
    graph: nx.DiGraph
) -> tuple[nx.DiGraph, Dict[str, str]]:
    """
    Relabel graph nodes using structural IDs instead of names.

    This ensures nodes are matched by their type and position in the workflow,
    not by their display names. The original name is preserved as a node attribute.

    Args:
        graph: Original graph with name-based node IDs

    Returns:
        Tuple of (relabeled_graph, mapping_dict) where:
        - relabeled_graph: Graph with structural IDs
        - mapping_dict: Maps new IDs back to original names
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function - not typically imported directly
from src.similarity import _relabel_graph_by_structure
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| graph || nx.DiGraph || Yes || Graph with name-based node IDs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| relabeled_graph || nx.DiGraph || Graph with structural IDs (trigger_N, node_N)
|-
| reverse_mapping || Dict[str, str] || Maps new IDs → original names
|}

=== Added Node Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| _original_name || str || Original display name of the node
|-
| _name_hash || int || Hash of normalized name for matching
|}

== Usage Examples ==

=== Relabeling Process ===
<syntaxhighlight lang="python">
import networkx as nx

# Original graph with user-defined names
G = nx.DiGraph()
G.add_node("My Email Trigger", type="n8n-nodes-base.gmailTrigger", is_trigger=True)
G.add_node("Send to Slack", type="n8n-nodes-base.slack", is_trigger=False)
G.add_node("Format Message", type="n8n-nodes-base.code", is_trigger=False)
G.add_edge("My Email Trigger", "Format Message")
G.add_edge("Format Message", "Send to Slack")

# Relabel
relabeled, mapping = _relabel_graph_by_structure(G)

# New node IDs:
# "My Email Trigger" → "trigger_0"
# "Send to Slack" → "node_0" (or node_1 based on sorting)
# "Format Message" → "node_1" (or node_0)

print(list(relabeled.nodes()))
# ['trigger_0', 'node_0', 'node_1']

print(mapping)
# {'trigger_0': 'My Email Trigger', 'node_0': '...', 'node_1': '...'}
</syntaxhighlight>

=== Sorting Logic ===
<syntaxhighlight lang="python">
def node_sort_key(node_tuple):
    name, data = node_tuple
    return (
        data.get("type", ""),        # Primary: by type string
        -graph.out_degree(name),      # Secondary: higher out-degree first
        -graph.in_degree(name),       # Tertiary: higher in-degree first
        name,                          # Final: alphabetical for determinism
    )

# This ensures:
# 1. Nodes of the same type are grouped
# 2. Hub nodes (high connectivity) come first
# 3. Results are deterministic
</syntaxhighlight>

=== Name Hash Normalization ===
<syntaxhighlight lang="python">
# Smart quotes are normalized for comparison
# U+2018 (') → U+0027 (')
# U+2019 (') → U+0027 (')
# U+201C (") → U+0022 (")
# U+201D (") → U+0022 (")

normalized_name = original_name.replace("\u2018", "'").replace("\u2019", "'")
normalized_name = normalized_name.replace("\u201c", '"').replace("\u201d", '"')
relabeled.nodes[new_label]["_name_hash"] = hash(normalized_name)

# This helps match nodes that differ only in quote style
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Graph_Relabeling]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
